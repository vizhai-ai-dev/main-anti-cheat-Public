# Docker Deployment Guide

## Overview

This guide covers deploying the AI Proctoring System using Docker and Docker Compose for both development and production environments.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 4+ CPU cores recommended

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository>
cd FINAL
```

### 2. Build and Run (API Only)
```bash
# Build and start API service only
docker-compose up -d proctoring-api

# Check status
docker-compose ps

# View logs
docker-compose logs -f proctoring-api
```

### 3. Build and Run (Full Stack)
```bash
# Build and start all services
docker-compose up -d

# Services will be available at:
# - API: http://localhost:5000
# - Streamlit UI: http://localhost:8501
# - Nginx (with SSL): https://localhost
# - Nginx (HTTP): http://localhost:8080
```

## Service Architecture

### Core Services

1. **proctoring-api** (Port 5000)
   - Flask REST API server
   - All AI proctoring modules
   - File upload handling
   - Health checks

2. **proctoring-ui** (Port 8501)
   - Streamlit web interface
   - Live monitoring
   - Module testing
   - Comprehensive analysis

3. **redis** (Port 6379)
   - Caching layer
   - Session storage
   - Optional service

4. **nginx** (Ports 80, 443, 8080)
   - Reverse proxy
   - Load balancing
   - SSL termination
   - Rate limiting

## Configuration Options

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
MAX_FILE_SIZE=100MB

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Model Configuration
YOLO_WEIGHTS_URL=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights
```

### Resource Limits

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## Deployment Scenarios

### 1. Development Environment

```bash
# Start API only for development
docker-compose up -d proctoring-api redis

# Or start with UI for testing
docker-compose up -d proctoring-api proctoring-ui redis
```

### 2. Production Environment

```bash
# Full production stack with Nginx
docker-compose up -d

# Or without Nginx if using external load balancer
docker-compose up -d proctoring-api proctoring-ui redis
```

### 3. API-Only Deployment

```bash
# For React/external frontend integration
docker-compose up -d proctoring-api redis
```

## Volume Management

### Persistent Data

```bash
# Create named volumes
docker volume create proctoring_models
docker volume create proctoring_uploads
docker volume create proctoring_redis

# Backup volumes
docker run --rm -v proctoring_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v proctoring_uploads:/data -v $(pwd):/backup alpine tar xzf /backup/uploads_backup.tar.gz -C /data
```

### Model Files

Models are automatically downloaded during container build. To use custom models:

```bash
# Copy models to host directory
mkdir -p ./models
cp your_custom_model.weights ./models/

# Models will be mounted to containers
```

## Monitoring and Logging

### Health Checks

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost:5000/health

# Streamlit health check
curl http://localhost:8501/_stcore/health
```

### Logs

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f proctoring-api

# View last 100 lines
docker-compose logs --tail=100 proctoring-api
```

### Monitoring

```bash
# Resource usage
docker stats

# Container inspection
docker inspect ai-proctoring-api

# Network inspection
docker network ls
docker network inspect final_proctoring-network
```

## Scaling

### Horizontal Scaling

```bash
# Scale API service
docker-compose up -d --scale proctoring-api=3

# Update Nginx upstream configuration for load balancing
```

### Vertical Scaling

Update resource limits in `docker-compose.yml`:

```yaml
proctoring-api:
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: '4.0'
```

## Security

### SSL/TLS Configuration

1. **Generate SSL certificates:**
```bash
mkdir ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem
```

2. **Update nginx.conf:**
```nginx
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

### Network Security

```bash
# Create custom network with isolation
docker network create --driver bridge proctoring-secure

# Update docker-compose.yml to use custom network
```

### Container Security

- Containers run as non-root user
- Minimal base images used
- Regular security updates
- Resource limits enforced

## Troubleshooting

### Common Issues

1. **Port conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :5000

# Change ports in docker-compose.yml
```

2. **Memory issues:**
```bash
# Check available memory
free -h

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
```

3. **Model download failures:**
```bash
# Manual model download
docker-compose exec proctoring-api wget -O /app/models/yolov3.weights \
  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights
```

4. **Permission issues:**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./models ./uploads
```

### Debug Mode

```bash
# Run with debug logging
FLASK_DEBUG=1 docker-compose up

# Access container shell
docker-compose exec proctoring-api bash

# Check container logs
docker-compose logs proctoring-api | grep ERROR
```

## Backup and Recovery

### Full System Backup

```bash
#!/bin/bash
# backup.sh

# Stop services
docker-compose down

# Backup volumes
docker run --rm -v proctoring_uploads:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/uploads_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

docker run --rm -v proctoring_redis:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/redis_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# Backup configuration
cp docker-compose.yml nginx.conf .env backups/

# Restart services
docker-compose up -d
```

### Recovery

```bash
#!/bin/bash
# restore.sh

# Stop services
docker-compose down

# Restore volumes
docker run --rm -v proctoring_uploads:/data -v $(pwd)/backups:/backup alpine \
  tar xzf /backup/uploads_YYYYMMDD_HHMMSS.tar.gz -C /data

# Restart services
docker-compose up -d
```

## Performance Optimization

### Image Optimization

```bash
# Multi-stage build for smaller images
# Use .dockerignore to exclude unnecessary files
# Use specific package versions
```

### Runtime Optimization

```bash
# Use production WSGI server
pip install gunicorn

# Update Dockerfile CMD
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "api_server:app"]
```

### Caching

```bash
# Enable Redis caching
docker-compose up -d redis

# Configure application to use Redis
```

## Production Checklist

- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Resource limits configured
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Load balancing configured
- [ ] Health checks enabled
- [ ] Logging centralized
- [ ] Documentation updated

## Support

For issues with Docker deployment:

1. Check logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `docker-compose exec proctoring-api curl localhost:5000/health`
4. Review resource usage: `docker stats` 