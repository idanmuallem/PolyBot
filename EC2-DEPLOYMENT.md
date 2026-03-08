# EC2 Deployment Guide

## One-Time Setup: Fix DB Path Issue

If your EC2 instance has `/home/ec2-user/trades.db` as a **directory** instead of a **file**, run this once:

```bash
# SSH into EC2
ssh ec2-user@your-instance

# Download and run the fix script
curl -O https://raw.githubusercontent.com/YOUR_REPO/main/fix-db-path.sh
chmod +x fix-db-path.sh
sudo ./fix-db-path.sh
```

Or run manually:

```bash
# 1. Stop the bot
docker stop polybot && docker rm polybot

# 2. Move your actual database file safely out of the weird folder
mv /home/ec2-user/trades.db/trades.db /home/ec2-user/actual_trades.db

# 3. Delete the mistaken directory Docker created
rm -rf /home/ec2-user/trades.db

# 4. Rename your backed-up file to the correct name
mv /home/ec2-user/actual_trades.db /home/ec2-user/trades.db

# 5. Set correct permissions
chmod 666 /home/ec2-user/trades.db

# Verify
ls -lh /home/ec2-user/trades.db
# Should show: -rw-rw-rw- 1 ec2-user ec2-user 128K ... /home/ec2-user/trades.db
```

## Environment Variables

Create `/home/ec2-user/.env` with your production secrets:

```bash
# Polymarket API
POLYMARKET_API_KEY=your_key_here
CLOB_ENDPOINT=https://clob.polymarket.com

# FRED Economic Data (optional)
FRED_API_KEY=your_fred_key

# Trading Config
MIN_EV=0.05
DRY_RUN=false
WATCH_ONLY=false
```

## Manual Deployment

```bash
# SSH into EC2
ssh ec2-user@your-instance

# Pull latest image
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 979667333506.dkr.ecr.eu-north-1.amazonaws.com
docker pull 979667333506.dkr.ecr.eu-north-1.amazonaws.com/polybot:latest

# Stop old container
docker stop polybot || true
docker rm polybot || true

# Run new container
docker run -d \
  --name polybot \
  -p 80:8501 \
  -v /home/ec2-user/trades.db:/app/trades.db \
  --env-file /home/ec2-user/.env \
  979667333506.dkr.ecr.eu-north-1.amazonaws.com/polybot:latest

# Check logs
docker logs -f polybot

# Check DB is working
docker exec polybot ls -lh /app/trades.db
# Should show: -rw-r--r-- 1 root root 128K ... /app/trades.db (file, not directory)
```

## Verification

```bash
# Check container status
docker ps | grep polybot

# Check DB path inside container
docker exec polybot sh -c "ls -lh /app && sqlite3 /app/trades.db 'SELECT COUNT(*) FROM hunt_history'"

# Access dashboard (if port 80 is open)
curl http://localhost

# Or from your browser
http://YOUR_EC2_PUBLIC_IP
```

## Troubleshooting

### DB is a directory instead of a file
Run the fix script above.

### Container won't start
```bash
docker logs polybot
# Check for permission/path errors
```

### Dashboard not accessible
```bash
# Check security group allows inbound port 80
# Check container is listening
docker exec polybot netstat -tlnp | grep 8501
```

### DB writes failing
```bash
# Check mount is correct
docker inspect polybot | grep -A 10 Mounts

# Check host file permissions
ls -lh /home/ec2-user/trades.db
sudo chmod 666 /home/ec2-user/trades.db
```
