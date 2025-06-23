# 🚀 Quick Start Guide - MLOps Project

## ⚡ **Start Everything (1 Command)**
```bash
cd "/Users/sergiolebedwright/Desktop/Master IE/ML Ops/mlops_group8-main"
docker run -d -p 8000:8000 --name mlops-demo mlops-group8-simple
```

## 🌐 **Access Points**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Base API**: http://localhost:8000

## 📋 **Test JSON (Copy & Paste into FastAPI docs)**

### Single Prediction:
```json
{
  "track_name": "How to Save a Life",
  "year": 2007,
  "artist_popularity": 67.0,
  "genre": "rock",
  "danceability": 0.64,
  "energy": 0.743,
  "liveness": 0.101,
  "loudness": 61.0,
  "speechiness": 0.0379,
  "tempo": 122.0,
  "valence": 0.361,
  "duration_ms": 262533.0,
  "acousticness": 0.269,
  "key": 10.0
}
```

### Different Success Bucket:
```json
{
  "track_name": "Experimental Track",
  "year": 2020,
  "artist_popularity": 30.0,
  "genre": "indie",
  "danceability": 0.4,
  "energy": 0.3,
  "liveness": 0.2,
  "loudness": 80.0,
  "speechiness": 0.15,
  "tempo": 90.0,
  "valence": 0.3,
  "duration_ms": 300000.0,
  "acousticness": 0.8,
  "key": 2.0
}
```

## 🎯 **Expected Results**
- **"How to Save a Life"**: `"has a great chance! It has an expected success of 67.2%!"`
- **"Experimental Track"**: `"will maybe be a hit! It has an expected success of 56.3%!"`

## 🔧 **Useful Commands**
```bash
# Check if container is running
docker ps

# Stop container
docker stop mlops-demo

# View container logs
docker logs mlops-demo

# Remove container
docker rm mlops-demo
```

## 📁 **Key Files**
- `app/main.py` - ✅ Fixed FastAPI code
- `test_prediction_with_metadata.json` - Single test
- `test_batch_with_metadata.json` - Batch test
- `PROJECT_SUMMARY_FINAL.md` - Complete documentation

## ✅ **Verification Checklist**
1. ⬜ Container starts without errors
2. ⬜ Health endpoint returns "ok"
3. ⬜ Predictions are 60-80% range (not 300+%)
4. ⬜ Messages show success buckets
5. ⬜ Both single and batch work

**Everything is saved and ready for tomorrow! 🎉**
