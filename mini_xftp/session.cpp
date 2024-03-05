#include "session.h"

sessionBase::~sessionBase()
{
    this->close();
}


int sessionBase::init()
{
    int libssh2_error  = libssh2_init(0);
    if (libssh2_error) {
        qDebug() << "libssh2_init error;" <<  libssh2_error;
        return -1;
    }
    _socket = new QTcpSocket();
    _socket->connectToHost(host_ip, port);
    if (!_socket->waitForConnected()) {
        qDebug() << "error connectting to host ip:" << host_ip << "user_name:" << username << "pass_word:" << passwd;
        return -2;
    }

    _session = libssh2_session_init();
    if (!_session) {
        qDebug() << "libssh2_session_init() failed";
        return -3;
    }
    libssh2_error = libssh2_session_startup(_session, _socket->socketDescriptor());
    if (libssh2_error) {
        qDebug() << "libssh2_session_startup error:" << libssh2_error;
        return -4;
    }
    libssh2_userauth_list(_session, username.toLocal8Bit().constData(), username.toLocal8Bit().length());
    if (libssh2_userauth_password(
                _session,
                username.toLocal8Bit().constData(),
                passwd.toLocal8Bit().constData()
                )) {
        qDebug() << "Passwd authentication failed";
        _socket->disconnectFromHost();
        libssh2_session_disconnect(_session, "Client disconnecting for error");
        libssh2_session_free(_session);
        libssh2_exit();
        return -5;
    }
    do {
        _sftp_session = libssh2_sftp_init(_session);
        if (!_sftp_session) {
            if (libssh2_session_last_errno(_session) == LIBSSH2_ERROR_EAGAIN) {
                qDebug() << "non-blocking init, wait for read" << endl;
                _socket->waitForReadyRead();
            } else {
                qDebug() << "unable to init SFTP session" << endl;
                return -6;
            }
        }
    } while (!_sftp_session);

    return 0;
}


int sessionBase::close()
{
    if (_sftp_session)
        libssh2_sftp_shutdown(_sftp_session);
    libssh2_session_disconnect(_session,"Client disconnecting normally");
    libssh2_session_free(_session);
    libssh2_exit();
    if (_socket->isOpen())
        _socket->disconnectFromHost();
    delete _socket;
    _socket = nullptr;
    return 0;
}


sessionIO::sessionIO(QString host_ip, QString username, QString passwd, int port, QObject *parent):
    QObject(parent),sessionBase(host_ip,username,passwd,port)
{

}


void sessionIO::download(QString local_file, QString remote_file)
{
    // download finished
    if(local_file=="" && remote_file=="") {
        emit finished(0);  // send a signal tell main thread downlod finished
        return;
    }
    QFile a_file(local_file);
    if (!a_file.open(QIODevice::WriteOnly)) {
        qDebug() << "can not open the file :" << local_file;
        return;
    }
    LIBSSH2_SFTP_HANDLE* sftp_handle;

    do {
        sftp_handle = libssh2_sftp_open(_sftp_session,remote_file.toUtf8().data(),LIBSSH2_FXF_READ,0);
        if (!sftp_handle)
        {
            //LIBSSH2_ERROR_EAGAIN 标记为非阻塞I/O,但调用将阻塞
            if (libssh2_session_last_errno(_session) != LIBSSH2_ERROR_EAGAIN)
            {
                qDebug() << "Unable to open file with ："<<libssh2_session_last_errno(_session)<<endl;
                libssh2_sftp_close(sftp_handle);
                return;
            }
            else
            {
                qDebug() << "non-blocking open";
                _socket->waitForReadyRead();
            }
        }
    } while (!sftp_handle);
    int rc;
    QByteArray byte_array;
    byte_array.resize(4096);
    // 返回一个指向QByteArray数据的指针，可以用于访问和修改QByteArray中的数据
    char* buffer = byte_array.data();
    //返回字节数组的大小
    int buffer_size = byte_array.size();
    int cnt = 0;
    do {
        do {
            rc = libssh2_sftp_read(sftp_handle,buffer,buffer_size);
            if (rc > 0) a_file.write(buffer,rc);
            // qDebug() << cnt <<endl;
            if(++cnt%_step==0) {
                emit process(_step*4096);
            }
        } while (rc > 0);
        emit process(cnt%_step*4096);
        if (rc  != LIBSSH2_ERROR_EAGAIN) {
            // qDebug() << "finished";
            break;
        }
    } while (true);
    qDebug() << remote_file << "----" << local_file<<endl;
    libssh2_sftp_close(sftp_handle);
    a_file.close();
}


void sessionIO::initSessionIO()
{
    this->sessionBase::init();
}


session::session(QString host_ip, QString username, QString passwd, int port, QObject *parent):
    QObject(parent),sessionBase(host_ip,username,passwd,port)
{
    this->sessionBase::init();
    _sessio = new sessionIO(host_ip,username,passwd,port);
    pThread = new QThread(this);
    _sessio->moveToThread(pThread);
    connect(this,&session::initSessionIO,_sessio,&sessionIO::initSessionIO);
    connect(this,&session::download,_sessio,&sessionIO::download);
    connect(_sessio,&sessionIO::process, this,&session::process);
    pThread->start();
    emit initSessionIO();
}


session::~session()
{
    pThread->quit();
    pThread->wait();
    delete pThread;
    delete _sessio;
}


char session::testFile(QString file)// d 代表文件夹 ， f 代表文件， 0 代表未识别
{
    QString cmd = QString::asprintf("[[ -d '%s' ]] && echo d || [[ -f '%s' ]] && echo f || echo 0;",file.toStdString().c_str(),file.toStdString().c_str());
    execSSH(cmd);
    return sshBuf[0];
}


void session::uploadFiles(QString file, QString remote_file)
{
    if(!QFile::exists(file)){
        qDebug()<<"path not exist: "<<file;
        return;
    }
    if(!QFileInfo(file).isDir()) {
        uploadFileViaSftp(file, remote_file);
        return;
    }
    execSSH(QString::asprintf("mkdir '%s'",remote_file.toUtf8().constData()));
    QDir dir(file);
    QStringList fileList=dir.entryList(QDir::Files);
    for(auto& item:fileList) {
        uploadFileViaSftp(dir.absoluteFilePath(item), remote_file+"/"+item);
    }
    QStringList dirList=dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for(auto& item:dirList) {
        uploadFiles(dir.absoluteFilePath(item), remote_file+"/"+item);
    }
}


void session::downloadFiles(QString file, QString remote_source_file)
{
    char c = testFile(remote_source_file);
    if(c=='0')
    {
        qDebug()<<"unknown file type"<<remote_source_file<<endl;
        return;
    }
    execSSH(QString::asprintf("du -s '%s' | cut -f 1", remote_source_file.toUtf8().constData()));
    _finished=0;
    _total = QString(sshBuf).toInt();  // KB
    _downloadFiles(file,remote_source_file);
    // Sleep(1000);
    emit download("",""); // 向下载线程发送下载完毕的信号
}


QString session::execSSH(QString command)
{
    if(!_session){
        qDebug() << "execSSH: session closed"<<endl;
        return "";
    }
    _channel = libssh2_channel_open_session(_session);
    for(int i=0;i<10&&!_channel;++i) {
        sessionBase::init();
        _channel = libssh2_channel_open_session(_session);
        qDebug()<<"session closed, reopening round "<<i+1<<endl;
    }
    if(!_channel){
        qDebug()<<"Unable to open a session";
        return "";
    }
    // 执行命令
    QString charset = "[[ `locale -a | grep 'zh_CN.utf8'` != '' ]] && export LANG='zh_CN.UTF-8';[[ `locale -a | grep 'C.utf8'` != '' ]] && export LANG='C.UTF-8';";
    // qDebug() << command << endl;
    libssh2_channel_exec(_channel, (charset+command).toStdString().c_str());
    // 读取命令输出
    memset(sshBuf,0,BUF_SIZE);
    libssh2_channel_read(_channel, sshBuf, BUF_SIZE);
    // qDebug()<<sshBuf<<endl;
    // 退出命令
    libssh2_channel_send_eof(_channel);
    // 关闭 shell 会话
    libssh2_channel_free(_channel);
    return QString(sshBuf);
}


void session::process(int x)
{

    _finished += x;
    emit updateBar(_finished*100/_total/1024);
}


int session::close()
{
    this->sessionBase::close();
    pThread->quit();
    pThread->wait();
    delete pThread;
    pThread = nullptr;
    delete _sessio;
    _sessio = nullptr;
    return 0;
}


int session::uploadFileViaSftp(const QString &local_filename, const QString &remote_filename)
{
    QFile local_file(local_filename);
    if (!local_file.open(QIODevice::ReadOnly)) {
        qDebug() << "can not open the file :" << local_filename <<QIODevice::ReadOnly;
        return -1;
    }

    // 创建远程文件
    LIBSSH2_SFTP_HANDLE *sftp_handle = libssh2_sftp_open(_sftp_session, remote_filename.toUtf8().data(),
                             LIBSSH2_FXF_WRITE | LIBSSH2_FXF_CREAT | LIBSSH2_FXF_TRUNC,
                             LIBSSH2_SFTP_S_IRWXU | LIBSSH2_SFTP_S_IRWXG | LIBSSH2_SFTP_S_IRWXO);

    if(!sftp_handle){
        qDebug() << "sftp_handle open failed";
        return 2;
    }

    // 读取本地文件并写入远程文件
    char buffer[4096];
    size_t nbytes;
    while ((nbytes = local_file.read(buffer, sizeof(buffer))) > 0) {
        libssh2_sftp_write(sftp_handle, buffer, nbytes);
    }
    qDebug()<<local_filename<<"----"<<remote_filename<<endl;
    // 关闭文件和会话
    local_file.close();
    libssh2_sftp_close(sftp_handle);
    return 0;
}


int session::_downloadFiles(QString file, QString remote_source_file)
{
    char c = testFile(remote_source_file);
    if(c=='0')
    {
        qDebug()<<"unknown file type"<<remote_source_file<<endl;
        return -1;
    }
    else if(c=='d')
    {
        if(mkdir(file.toLocal8Bit().constData())==-1)
        {
            qDebug()<<file<<"already exists and it will be coverd!"<<endl;
            remove(file.toLocal8Bit().constData());
            mkdir(file.toLocal8Bit().constData());
        }
        QString cmd = QString::asprintf("ls '%s'",remote_source_file.toStdString().c_str());
        execSSH(cmd);
        std::stringstream inputStream(sshBuf);
        std::string filename;
        while(std::getline(inputStream,filename,'\n'))
        {
            QString newLocalFile = file+"/"+QString::fromStdString(filename);
            QString newRemoteFile = remote_source_file+"/"+QString::fromStdString(filename);
            _downloadFiles(newLocalFile,newRemoteFile);
        }
    }
    else if(c=='f')
    {
        emit download(file,remote_source_file);
    }
    else
    {
        qDebug()<<"test file loaded:"<<remote_source_file<<endl;
        return -2;
    }
    return 0;
}










