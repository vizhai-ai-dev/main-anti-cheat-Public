#!/bin/bash
# Start the backend server
echo "Starting the backend server..."
python backend_server.py &
BACKEND_PID=$!

# Wait for the backend to start
sleep 2

# Start the frontend server
echo "Starting the frontend server..."
cd vizhAI && npm start

# When frontend is terminated, kill the backend
kill $BACKEND_PID 