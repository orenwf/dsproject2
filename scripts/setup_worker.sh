#!/bin/bash

# need to have master ssh key on worker
# stop all nodes
/usr/local/spark/sbin/stop-all.sh
MASTER_IP=$(hostname -I | awk '{print $01}')
# copy the setup scripts across
scp -r -o CheckHostIP=no scripts/ root@$1:scripts
# add the slave ip to the master
echo "$1" >> /usr/local/spark/conf/slaves
# then ssh into the worker and run the setup_spark.sh script
ssh -o CheckHostIP=no root@$1 "chmod +x scripts/* && \\
./scripts/setup_spark.sh $MASTER_IP && \\
systemctl reboot"

