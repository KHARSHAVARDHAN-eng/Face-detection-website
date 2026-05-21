# FaceAuth AI

A full-stack AI-powered Face Recognition Web Application.

## Architecture
- **Frontend**: React, Vite, Tailwind CSS, `react-hot-toast`
- **Backend**: FastAPI (Python), SQLAlchemy (SQLite)
- **AI Engine**: `face_recognition`, OpenCV, `dlib`

---

## 🤖 Mock vs. Real AI Mode
The application can run in two modes controlled by the `MOCK_AI_MODE` environment variable in the backend `.env` file:
- **Mock Mode (`MOCK_AI_MODE=true`)**: Falls back to deterministic image hashing instead of real face recognition. Ideal for testing UI, API flows, and deployment without dealing with complex `dlib` C++ compilation.
- **Real Mode (`MOCK_AI_MODE=false`)**: Uses `dlib` and OpenCV to extract genuine 128-d face embeddings and calculates Euclidean distance for recognition.

---

## 🐳 Docker Deployment (Recommended)
You can run both the frontend and backend easily using Docker Compose.

1. Create a `.env` file in the `backend/` directory:
   ```env
   MOCK_AI_MODE=true
   CORS_ORIGINS=*
   ```
2. Run the application:
   ```bash
   docker-compose up --build -d
   ```
- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Manual Setup

### Backend
1. Navigate to `backend/` and create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend
1. Navigate to `frontend/`:
   ```bash
   cd frontend
   ```
2. Create `.env`:
   ```env
   VITE_API_URL=http://localhost:8000/api
   VITE_STATIC_URL=http://localhost:8000
   ```
3. Install dependencies and run:
   ```bash
   npm install
   npm run dev
   ```

---

## 🛠 Troubleshooting `dlib` on macOS (Apple Silicon)
Installing `face_recognition` requires compiling `dlib`, which can fail on M1/M2/M3 chips due to missing `<fp.h>` headers (associated with `libpng`).

**Workarounds:**
1. Use the pre-configured **Mock AI Mode**.
2. Alternatively, compile `dlib` manually by disabling PNG support:
   ```bash
   brew install cmake
   CMAKE_ARGS="-DDLIB_PNG_SUPPORT=OFF -DDLIB_JPEG_SUPPORT=OFF" pip install --no-cache-dir dlib
   pip install face_recognition
   ```

---

## 🚀 Cloud Deployment Guides

### Frontend (Vercel)
1. Push your repository to GitHub.
2. Go to Vercel, import your repository, and select the `frontend` root directory.
3. Add Environment Variables:
   - `VITE_API_URL`: Your deployed backend URL + `/api`
   - `VITE_STATIC_URL`: Your deployed backend URL
4. Deploy!

### Backend (Render)
1. Go to Render.com and create a new **Web Service**.
2. Connect your GitHub repository.
3. Configure settings:
   - **Root Directory**: `backend`
   - **Environment**: Docker (Render will automatically detect the `backend/Dockerfile`)
4. Add Environment Variables:
   - `MOCK_AI_MODE=true`
   - `CORS_ORIGINS=https://your-vercel-frontend-url.vercel.app`
5. Deploy!

---

## 📖 API Usage Examples
The API comes with an interactive Swagger UI at `/docs`.

### Register User
```bash
curl -X POST "http://localhost:8000/api/register" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "name=John Doe" \
  -F "image=@capture.jpg"
```

### Recognize Face
```bash
curl -X POST "http://localhost:8000/api/recognize" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@capture.jpg"
```
