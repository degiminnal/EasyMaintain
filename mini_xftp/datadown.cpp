#include "datadown.h"
#include "ui_datadown.h"


#include<iostream>
#include <string>
#include <sstream>
#include<qfont.h>
#include<qapplication.h>
#include <QFileInfo>
#include <QDir>
#include <QIcon>
#include <libssh/libssh2_config.h>
#include<QTextCodec>


using namespace std;
/*
1. 进度条没实现
2.指针类型转换 需要修改
3. 4个模块 只实现了RRM,其余还没实现
*/


DataDown::DataDown(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DataDown)
{
    ui->setupUi(this);
    sess = nullptr;
    resize(800,800);

    ui->IPLlineEdit->setText("192.168.3.6");
    ui->PortLineEdit->setText("22");
    ui->NameLineEdit->setText("kylin");
    ui->PWLineEdit->setText("qwe123");

    RRM = new QRadioButton("RRM");
    BCN = new QRadioButton("BCN");
    PPN = new QRadioButton("PPN");
    TRK = new QRadioButton("TRK");

    changeLocalDir(".");

}

DataDown::~DataDown()
{
    delete ui;
    delete sess;
}

int DataDown::init()
{
    QString host_ip = ui->IPLlineEdit->text();
    int port = ui->PortLineEdit->text().toInt();
    QString user_name = ui->NameLineEdit->text();
    QString pass_word = ui->PWLineEdit->text();
    sess = new session(host_ip, user_name, pass_word, port);
    connect(sess, &session::updateBar,this, &DataDown::updateBar);
    // things to do after downfinished
    connect(sess->_sessio, &sessionIO::finished,this,[=](int x){
        sess->_finished = sess->_total;
        updateBar(100);
        if(x==0) changeLocalDir(".");
        else if(x==1) changeRemoteDir(".");
        else qDebug() << "finished(int x) accept only 0 or 1 as para!" << endl;
    });
    return 1;
}

int DataDown::close()
{
    if(sess && sess->pThread->isRunning())
    {
        sess->close();
        sess = nullptr;
        qDebug() << "closed session"<<endl;
        return 0;
    }
    qDebug() << "session is not opened"<<endl;
    return 1;
}

int DataDown::changeRemoteDir(QString filename)
{
    if(!sess){
        qDebug() << "changeRemoteDir: session closed"<<endl;
        return -1;
    }
    QString line;
    QVector <QString> files;
    if(filename==".."){
        if(pwdRemote!="/"){
            int x = pwdRemote.size() - pwdRemote.section('/',-1,-1).size();
            while(x>1&&pwdRemote[x-1]=='/') --x;
            pwdRemote = pwdRemote.mid(0,x);
        }
    }
    else if(filename!=".") {
        if(pwdRemote!="/") pwdRemote+="/";
        pwdRemote+=filename;
    }
    qDebug()<<"current pwdRemote: "<<pwdRemote<<endl;
    if(sess->testFile(pwdRemote)!='d'){
        qDebug()<<pwdRemote<<" is not a dir!"<<endl;
        changeRemoteDir("..");
        return -2;
    }
    QString cmd = QString::asprintf("ls %s -laQ",pwdRemote.toUtf8().constData());
    QTextStream inputStream(sess->execSSH(cmd).toLocal8Bit().constData());
    // exclude "total xxx" and "."
    while (inputStream.readLineInto(&line))
        if(line.endsWith(".\"")) break;
    while (inputStream.readLineInto(&line)){
        // exclude link file
        line = line.mid(0, line.size()-1);
        if(line[0]=='l') continue;
        // '0' for folder and '1' for file at the 1st position
        files.push_back((line[0]=='d'?QString("0"):QString("1"))+line.split("\"").last());
    }
    sort(files.begin(), files.end(), less<QString>());
    updateListWidget(files, ui->RemoteListWidget);
    ui->RemoteFileDir->setText(pwdRemote);
    return 0;
}

int DataDown::changeLocalDir(QString filename)
{
    if(filename==".."){
        if(pwdLocal.size()>3){
            int x = pwdLocal.size() - pwdLocal.section('/',-2,-2).size() - 1;
            pwdLocal = pwdLocal.mid(0,x);
        }
        else if(pwdLocal.size()==3) {
            pwdLocal = "/";
        }
    }
    else if(filename!=".") {
        if(pwdLocal!="/") pwdLocal+=filename+"/";
        else pwdLocal=filename;
    }
    qDebug()<<"current pwdLocal: "<<pwdLocal<<endl;
    QVector<QString> files={"0.."};
    if(pwdLocal=="/"){
        foreach (QFileInfo file, QDir::drives()) files.push_back(QString("0")+file.absoluteFilePath());
    }
    else {
        QDir dir(pwdLocal);
        QFileInfoList file_list = dir.entryInfoList(QDir::Files | QDir::Hidden | QDir::NoSymLinks);
        QFileInfoList folder_list = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
        for(auto& file:folder_list) files.push_back(QString("0")+file.fileName());
        for(auto& file:file_list) files.push_back(QString("1")+file.fileName());
        sort(files.begin(), files.end());
    }
    updateListWidget(files, ui->LocalListWidget);
    ui->LocalFileDir->setText(pwdLocal);

    return 0;
}

int  DataDown::updateListWidget(QVector<QString>& files, QListWidget* listWiget) {
    listWiget->clear();
    for(QString& file:files){
        QListWidgetItem *item = new QListWidgetItem(file.mid(1));
        if(file[0]=='0') item->setForeground(QColor(100, 100, 100));
        listWiget->addItem(item);
    }
    // Sleep(100);   // 刷新太快会卡死，设置了500ms的延迟
    return 0;
}

void DataDown::on_RRMRadio_clicked()
{
    ui->textEdit->append("RRM!");
}

void DataDown::on_BCNRadio_clicked()
{
  ui->textEdit->append("BCN!");
}

void DataDown::on_TRKRadio_clicked()
{
  ui->textEdit->append("TRK!");
}

void DataDown::on_PPNRadio_clicked()
{
  ui->textEdit->append("PPN!");
}

void DataDown::on_connectBtn_clicked()
{
    this->close();
    if (init() != 1) {
        qDebug() << "Init session error!";
        return;
    }
    ui->textEdit->setText("连接服务器成功！");
    changeRemoteDir(QString("."));
}

void DataDown::on_disconnetBtn_clicked()
{
    this->close();
}

//    qDebug() << sess->pThread->isRunning();
//    QString filename = ui->LocalListWidget->currentItem()->text();
//    QString remote_file_path= ui->RemoteFileDir->text()+"/"+filename;
//    QString local_file_path= ui->LocalFileDir->text() + filename;
//    sess->uploadFiles(local_file_path, remote_file_path);
//    qDebug()<<"上传文件"<<endl;
//    changeRemoteDir(QString("."));  // 上传后刷新远程目录

void DataDown::on_downBtn_clicked()
{
    if(!sess || !sess->pThread->isRunning()){
        qDebug() << "sess is not established";
        return;
    }
    ui->progressBar->setValue(0);
    QString filename = ui->RemoteListWidget->currentItem()->text();
    QString remote_file_path= ui->RemoteFileDir->text()+"/"+filename;
    QString local_file_path= ui->LocalFileDir->text() + filename;
    sess->downloadFiles(local_file_path, remote_file_path);
    // changeLocalDir(".");  // 下载后刷新本地目录
}

void DataDown::on_LocalListWidget_itemDoubleClicked(QListWidgetItem *item)
{
        QString file = item->text();
        changeLocalDir(file);
}

void DataDown::on_RemoteListWidget_itemDoubleClicked(QListWidgetItem *item)
{
        QString file = item->text();
        changeRemoteDir(file);
}


void DataDown::updateBar(int x)
{
    ui->progressBar->setValue(x);
}

