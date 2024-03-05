#ifndef DATADOWN_H
#define DATADOWN_H

#include <QMainWindow>
#include <QGroupBox>
#include <QCheckBox>
#include <QKeySequenceEdit>
#include <QToolButton>
#include <QPushButton>
#include <QLabel>
#include <QMessageBox>
#include <QLineEdit>
#include <QList>
#include <QVector>
#include <QListWidget>

#include <QTcpServer>
#include <QTcpSocket>
#include <QNetworkInterface>
#include <QHostInfo>
#include <QNetworkAddressEntry>
#include <QHostAddress>
#include <QRadioButton>

#include <QDebug>
#include <QByteArray>

#include <QFile>
#include <QTextStream>
#include <QFileDialog>
#include <QNetworkAccessManager>
#include <libssh/libssh2.h>
#include <libssh/libssh2_sftp.h>
#include <vector>
#include <QThread>

#include "session.h"




namespace Ui {
class DataDown;
}

class DataDown : public QWidget
{
    Q_OBJECT

public:
    explicit DataDown(QWidget *parent = nullptr);
    void updateBar(int x);
    ~DataDown();

private slots:
    void on_RRMRadio_clicked();

    void on_BCNRadio_clicked();

    void on_TRKRadio_clicked();

    void on_PPNRadio_clicked();

    void on_connectBtn_clicked(); //连接服务器

    void on_disconnetBtn_clicked();  //断开服务器

    void on_downBtn_clicked(); //下载数据

    void on_LocalListWidget_itemDoubleClicked(QListWidgetItem *item);

    void on_RemoteListWidget_itemDoubleClicked(QListWidgetItem *item);

    void on_progressBar_valueChanged(int value);

private:
    Ui::DataDown *ui;
    session      *sess;

    QRadioButton *RRM;
    QRadioButton *BCN;
    QRadioButton *TRK;
    QRadioButton *PPN;

    QPushButton *connectBtn;
    QPushButton *disconnectBtn;

    QString _downloadFileName;

    QString pwdLocal = "E:/TestDownFile/";
    QString pwdRemote = "/home/kylin/test";
    QString buf;



    int init();

    int close();

    int changeRemoteDir(QString filename);
    int changeLocalDir(QString filename);
    int updateListWidget(std::vector<std::string>& files, QListWidget* listWiget);
    int updateListWidget(QVector<QString>& files, QListWidget* listWiget);


};

#endif // DATADOWN_H
