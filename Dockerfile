FROM ubuntu:latest

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    				    python3.8 \
    				    python3-pip \
    				    qt5-default \
    			 	    openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/* 

COPY . /NN_Without_Frameworks
WORKDIR /NN_Without_Frameworks
RUN pip3 install numpy matplotlib
RUN python3.8 python_nn/train_regression.py
RUN python3.8 python_nn/train_classification_numpy_nn.py

RUN mkdir -p cpp_nn/build
WORKDIR cpp_nn/build
RUN qmake ..
RUN make
RUN ./cpp_nn

WORKDIR /NN_Without_Frameworks/java_nn/src/ 
RUN javac train_regression.java
RUN java train_regression
