#!/bin/bash
# Fix EC2 DB path issue - Run this once on EC2 instance to correct the directory-mount problem

set -e

echo "=== Fixing PolyBot DB Path Issue ==="

# 1. Stop the bot
echo "Stopping polybot container..."
docker stop polybot 2>/dev/null || echo "Container not running"
docker rm polybot 2>/dev/null || echo "Container not found"

# 2. Check if /home/ec2-user/trades.db is a directory
if [ -d "/home/ec2-user/trades.db" ]; then
    echo "Found directory at /home/ec2-user/trades.db - fixing..."
    
    # 3. Move actual database file safely out
    if [ -f "/home/ec2-user/trades.db/trades.db" ]; then
        echo "Moving actual DB file to safe location..."
        mv /home/ec2-user/trades.db/trades.db /home/ec2-user/actual_trades.db
    else
        echo "No DB file found in directory, will create empty file"
    fi
    
    # 4. Delete the mistaken directory
    echo "Removing incorrect directory..."
    rm -rf /home/ec2-user/trades.db
    
    # 5. Restore DB as proper file
    if [ -f "/home/ec2-user/actual_trades.db" ]; then
        echo "Restoring DB file to correct location..."
        mv /home/ec2-user/actual_trades.db /home/ec2-user/trades.db
    else
        echo "Creating empty DB file..."
        touch /home/ec2-user/trades.db
    fi
    
    echo "✓ DB path fixed successfully"
else
    echo "✓ /home/ec2-user/trades.db is already a file (correct)"
fi

# 6. Ensure correct permissions
echo "Setting correct permissions..."
chmod 666 /home/ec2-user/trades.db

# 7. Verify
echo ""
echo "=== Verification ==="
ls -lh /home/ec2-user/trades.db

echo ""
echo "=== Fix Complete ==="
echo "You can now redeploy the container with the correct mount"
