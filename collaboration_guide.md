# Collaboration & Development Guide

This guide explains how you and your friends can collaborate, run the project locally, make changes, and push them to the shared GitHub repository.

---

## 🛠️ 1. Local Setup for Collaborators

Your friends should follow these steps to run the application on their own machines:

### Step 1: Clone the Repository
```bash
git clone https://github.com/KHARSHAVARDHAN-eng/Face-detection-website.git
cd Face-detection-website
```

### Step 2: Set Up the Backend
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   - **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `backend/` folder:
   ```env
   MOCK_AI_MODE=true
   CORS_ORIGINS=*
   ```
   *(Note: Setting `MOCK_AI_MODE=true` is highly recommended for collaborators to avoid having to compile C++ `dlib` libraries on their local machines).*
5. Run the FastAPI backend server:
   ```bash
   python main.py
   ```

### Step 3: Set Up the Frontend
1. Open a new terminal window/tab and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `frontend/` folder:
   ```env
   VITE_API_URL=/api
   VITE_STATIC_URL=
   ```
4. Run the Vite React frontend server:
   ```bash
   npm run dev
   ```

Now, the site will be accessible locally at `http://localhost:5173/`.

---

## 🔄 2. Git Collaboration Workflow

To avoid conflicts on the `main` branch, it is best practice to use **feature branches** and **Pull Requests**:

1. **Pull the latest changes** from the main branch:
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create a new branch** for the feature you are working on:
   ```bash
   git checkout -b feature/add-some-feature
   ```
3. **Make and test changes** locally.
4. **Stage and commit changes**:
   ```bash
   git add .
   git commit -m "Add description of your changes"
   ```
5. **Push your branch** to GitHub:
   ```bash
   git push -u origin feature/add-some-feature
   ```
6. **Open a Pull Request (PR)**:
   - Go to [GitHub Repository](https://github.com/KHARSHAVARDHAN-eng/Face-detection-website)
   - Click "Compare & pull request" next to your branch name.
   - Fill in details and click "Create pull request".
   - You can review and merge their code into the `main` branch.

---

## 🌐 3. Exposing Live Changes (Cloudflare Tunnels)

- **Harsha's Cloudflare Link**: The current link (`https://financing-deleted-filter-instead.trycloudflare.com`) points to **Harsha's local computer**.
- If Harsha pulls the friends' merged changes and runs the local server, those changes will show up on that link immediately.
- If a collaborator wants to host their *own* active workspace so others can see their changes, they can download the `cloudflared` binary and run:
  ```bash
  ./cloudflared tunnel --url http://localhost:5173
  ```
  This will output a unique, active `trycloudflare.com` URL pointing to their own local server instance.
