#!/bin/bash

apt-get update -y
apt-get upgrade -y
apt-get install -y default-jdk
echo 'export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")' >> .bashrc
echo 'export PATH=/usr/local/spark:$PATH' >> .bashrc
echo 'export PYSPARK_PYTHON=/usr/bin/python3' >> .bashrc
wget http://apache-mirror.8birdsvideo.com/spark/spark-3.0.0-preview/spark-3.0.0-preview-bin-hadoop3.2.tgz
tar xzvf spark-3.0.0-preview-bin-hadoop3.2.tgz
mv spark-3.0.0-preview-bin-hadoop3.2 /usr/local/spark
cp /usr/local/spark/conf/spark-env.sh.template /usr/local/spark/conf/spark-env.sh
echo "SPARK_LOCAL_IP=$(hostname -I | awk '{print $01}')" >> /usr/local/spark/conf/spark-env.sh 
echo "SPARK_MASTER_HOST=$1" >> /usr/local/spark/conf/spark-env.sh
echo "SPARK_PUBLIC_DNS=$1" >> /usr/local/spark/conf/spark-env.sh
