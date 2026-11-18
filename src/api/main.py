
from fastapi import FastAPI

# Initialize the FastAPI application
app = FastAPI(
    title="PathSnap API",
    description="GPS Map-Matching System",
    version="1.0.0"
)


# Home route
@app.get("/")
def home():
    return {
        "message": "Welcome to PathSnap API",
        "status": "running"
    }


# Health check route
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Backend"
    }

# Get available student-test trips
@app.get("/trips", tags=["Trips"])
def get_trips():
    trips = [f"trip_{i:03d}" for i in range(1, 11)]

    return {
        "total_trips": len(trips),
        "trips": trips
    }
