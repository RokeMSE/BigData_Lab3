# lowlevel_dt_reg.py
# Low-level Decision Tree Regression (RDD only) for NYC Taxi (train has label, test no label)
# Outputs predictions.csv with columns: id,y_pred

from pyspark.sql import SparkSession
from pyspark import StorageLevel
import math, sys, csv, argparse
from io import StringIO
from datetime import datetime

# -----------------------
# CLI args
# -----------------------
ap = argparse.ArgumentParser()
ap.add_argument("train_csv")
ap.add_argument("test_csv")
ap.add_argument("--maxDepth", type=int, default=5)
ap.add_argument("--numBins", type=int, default=16)
ap.add_argument("--minSamples", type=int, default=400)
args, _ = ap.parse_known_args()

# -----------------------
# Spark
# -----------------------
spark = SparkSession.builder.appName("LowLevelDTReg_NYCTaxi").getOrCreate()
sc = spark.sparkContext

# -----------------------
# Utils: CSV + datetime
# -----------------------
DT_FORMATS = ["%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]

def parse_dt(s):
    s = (s or "").strip()
    for fmt in DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

def split_csv(line):
    return next(csv.reader(StringIO(line)))

def get_idx(header_cols, name):
    # tìm chỉ số cột theo tên, không phân biệt hoa thường
    lc = [c.strip().lower() for c in header_cols]
    name = name.lower()
    try:
        return lc.index(name)
    except ValueError:
        return -1

def haversine_km(lat1, lon1, lat2, lon2):
    # khoảng cách great-circle (km)
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a + 1e-15))

# -----------------------
# Parse TRAIN (có label)
# -----------------------
def load_train(path):
    lines = sc.textFile(path)
    header = split_csv(lines.first())
    data = lines.filter(lambda r: r != ",".join(header))

    # indices
    ivendor   = get_idx(header, "vendor_id")
    idt       = get_idx(header, "pickup_datetime")
    ipass     = get_idx(header, "passenger_count")
    ilon1     = get_idx(header, "pickup_longitude")
    ilat1     = get_idx(header, "pickup_latitude")
    ilon2     = get_idx(header, "dropoff_longitude")
    ilat2     = get_idx(header, "dropoff_latitude")
    iflag     = get_idx(header, "store_and_fwd_flag")
    ilabel    = get_idx(header, "trip_duration")
    if min([ivendor, idt, ipass, ilon1, ilat1, ilon2, ilat2, iflag, ilabel]) < 0:
        raise ValueError("train.csv thiếu cột cần thiết.")

    def parse_row(line):
        try:
            c = split_csv(line)
            dt = parse_dt(c[idt]); 
            if dt is None: return None
            hour = float(dt.hour); dow = float(dt.weekday())

            vendor = float(c[ivendor]); psg = float(c[ipass])
            lon1 = float(c[ilon1]); lat1 = float(c[ilat1])
            lon2 = float(c[ilon2]); lat2 = float(c[ilat2])
            flag = 0.0 if str(c[iflag]).strip().upper() in ("N","0","","FALSE") else 1.0
            dist = haversine_km(lat1, lon1, lat2, lon2)

            y = float(c[ilabel])

            feats = [vendor, psg, lon1, lat1, lon2, lat2, flag, hour, dow, dist]
            return (feats, y)
        except Exception:
            return None

    rdd = data.map(parse_row).filter(lambda x: x is not None)
    return rdd

# -----------------------
# Parse TEST (không có label) -> (id, feats)
# -----------------------
def load_test(path):
    lines = sc.textFile(path)
    header = split_csv(lines.first())
    data = lines.filter(lambda r: r != ",".join(header))

    iid      = get_idx(header, "id")
    ivendor  = get_idx(header, "vendor_id")
    idt      = get_idx(header, "pickup_datetime")
    ipass    = get_idx(header, "passenger_count")
    ilon1    = get_idx(header, "pickup_longitude")
    ilat1    = get_idx(header, "pickup_latitude")
    ilon2    = get_idx(header, "dropoff_longitude")
    ilat2    = get_idx(header, "dropoff_latitude")
    iflag    = get_idx(header, "store_and_fwd_flag")
    if min([iid, ivendor, idt, ipass, ilon1, ilat1, ilon2, ilat2, iflag]) < 0:
        raise ValueError("test.csv thiếu cột cần thiết.")

    def parse_row(line):
        try:
            c = split_csv(line)
            _id = c[iid]
            dt = parse_dt(c[idt]); 
            if dt is None: return None
            hour = float(dt.hour); dow = float(dt.weekday())

            vendor = float(c[ivendor]); psg = float(c[ipass])
            lon1 = float(c[ilon1]); lat1 = float(c[ilat1])
            lon2 = float(c[ilon2]); lat2 = float(c[ilat2])
            flag = 0.0 if str(c[iflag]).strip().upper() in ("N","0","","FALSE") else 1.0
            dist = haversine_km(lat1, lon1, lat2, lon2)

            feats = [vendor, psg, lon1, lat1, lon2, lat2, flag, hour, dow, dist]
            return (_id, feats)
        except Exception:
            return None

    rdd = data.map(parse_row).filter(lambda x: x is not None)
    return rdd

# -----------------------
# Load data
# -----------------------
train_rdd = load_train(args.train_csv).repartition(256).persist(StorageLevel.MEMORY_AND_DISK)
test_id_feats = load_test(args.test_csv).repartition(128).persist(StorageLevel.MEMORY_AND_DISK)
train_rdd.count(); test_id_feats.count()
num_features = len(train_rdd.first()[0])
print("Train count =", train_rdd.count())
print("Test  count =", test_id_feats.count())

# -----------------------
# Stats & impurity
# -----------------------
def compute_stats(frdd):
    def seq(acc, x):
        y = x[1]
        return (acc[0] + 1, acc[1] + y, acc[2] + y*y)
    def comb(a,b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    return frdd.aggregate((0,0.0,0.0), seq, comb)

def sse(n,s,s2):
    return s2 - (s*s)/n if n > 0 else 0.0

def node_pred(n,s):
    return 0.0 if n == 0 else s / n

def feature_min_max(frdd, fi):
    def seq(acc, x):
        v = x[0][fi]
        return (min(acc[0], v), max(acc[1], v))
    def comb(a,b):
        return (min(a[0], b[0]), max(a[1], b[1]))
    return frdd.aggregate((float("inf"), float("-inf")), seq, comb)

# -----------------------
# Tree node
# -----------------------
class Node:
    __slots__ = ("prediction","fi","thr","left","right")
    def __init__(self, prediction, fi=None, thr=None, left=None, right=None):
        self.prediction = prediction
        self.fi = fi; self.thr = thr
        self.left = left; self.right = right
    def is_leaf(self): return self.left is None and self.right is None

# -----------------------
# Best split (simple threshold scan)
# -----------------------
def best_split(frdd, parent_sse, min_samples, num_bins=16):
    best = {"fi": None, "thr": None, "impurity": float("inf")}
    for fi in range(num_features):
        lo, hi = feature_min_max(frdd, fi)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo: 
            continue
        step = (hi - lo) / num_bins
        if step <= 0 or not math.isfinite(step): 
            continue
        thresholds = [lo + step*k for k in range(1, num_bins)]
        for thr in thresholds:
            left  = frdd.filter(lambda x, i=fi, t=thr: x[0][i] <= t)
            right = frdd.filter(lambda x, i=fi, t=thr: x[0][i]  > t)
            ln, ls, ls2 = compute_stats(left)
            rn, rs, rs2 = compute_stats(right)
            if ln < min_samples or rn < min_samples:
                continue
            imp = sse(ln,ls,ls2) + sse(rn,rs,rs2)
            if imp < best["impurity"]:
                best = {"fi": fi, "thr": thr, "impurity": imp}
    gain = parent_sse - best["impurity"]
    if best["fi"] is None or gain <= 1e-9:
        return None
    return best

# -----------------------
# Build tree
# -----------------------
def build_tree(frdd, depth, max_depth, min_samples, num_bins):
    frdd = frdd.persist(StorageLevel.MEMORY_AND_DISK)
    n, s, s2 = compute_stats(frdd)
    pred = node_pred(n, s)
    if depth >= max_depth or n < 2*min_samples:
        return Node(pred)
    parent = sse(n,s,s2)
    split = best_split(frdd, parent, min_samples, num_bins)
    if split is None:
        return Node(pred)
    fi, thr = split["fi"], split["thr"]
    left  = frdd.filter(lambda x, i=fi, t=thr: x[0][i] <= t)
    right = frdd.filter(lambda x, i=fi, t=thr: x[0][i]  > t)
    left.count(); right.count()  # materialize
    return Node(pred, fi, thr,
                build_tree(left,  depth+1, max_depth, min_samples, num_bins),
                build_tree(right, depth+1, max_depth, min_samples, num_bins))

def predict_one(f, node):
    cur = node
    while not cur.is_leaf():
        cur = cur.left if f[cur.fi] <= cur.thr else cur.right
    return cur.prediction

def rmse(frdd, tree):
    mse = frdd.map(lambda x: (x[1] - predict_one(x[0], tree))**2).mean()
    return math.sqrt(mse)

def mae(frdd, tree):
    return frdd.map(lambda x: abs(x[1] - predict_one(x[0], tree))).mean()

def r2(frdd, tree):
    n, s, s2 = compute_stats(frdd)  
    sst = s2 - (s*s)/n if n > 0 else float("nan")
    sse = frdd.map(lambda x: (x[1] - predict_one(x[0], tree))**2).sum()
    return 1 - sse/sst if (sst is not None and sst > 0) else float("nan")


# -----------------------
# Train & evaluate
# -----------------------
tree = build_tree(train_rdd, 0, args.maxDepth, args.minSamples, args.numBins)
train_rmse = rmse(train_rdd, tree)
print(f"Train RMSE: {train_rmse:.6f}")
train_mae = mae(train_rdd, tree)
train_r2  = r2(train_rdd, tree)
print(f"Train MAE:  {train_mae:.6f}")
print(f"Train R^2:  {train_r2:.6f}")

# -----------------------
# Predict TEST & write CSV (id,y_pred)
# -----------------------
pred_lines = (
    test_id_feats
    .map(lambda t: f"{t[0]},{predict_one(t[1], tree)}")  # "id,yhat"
)

header = sc.parallelize(["id,y_pred"])
out_dir = "predictions_csv"
header.union(pred_lines).coalesce(1).saveAsTextFile(out_dir)
print(f"Predictions saved to folder: {out_dir}")
print("Gộp thành 1 file duy nhất:", "cat predictions_csv/part-* > predictions.csv")

spark.stop()
