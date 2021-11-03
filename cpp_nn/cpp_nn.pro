TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        activations.cpp \
        initializers.cpp \
        layers.cpp \
        main.cpp \
        module.cpp \
        utils.cpp

HEADERS += \
    activations.h \
    initializers.h \
    layers.h \
    module.h \
    utils.h
