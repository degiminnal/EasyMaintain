#-------------------------------------------------
#
# Project created by QtCreator 2023-12-06T15:06:53
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = CK-degim
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        datadown.cpp \
        dataupdate/pingfunction.cpp \
        dataupdate/softupdate.cpp \
        fmain.cpp \
        main.cpp \
        session.cpp

HEADERS += \
        datadown.h \
        dataupdate/pingfunction.h \
        dataupdate/softupdate.h \
        fmain.h \
        libssh/libssh2.h \
        libssh/libssh2_config.h \
        libssh/libssh2_publickey.h \
        libssh/libssh2_sftp.h \
        session.h

FORMS += \
        datadown.ui \
        dataupdate/softupdate.ui \
        fmain.ui

LIBS += -L$$PWD/./lib -llibssh2

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    dataupdate/dataupdate.pri
