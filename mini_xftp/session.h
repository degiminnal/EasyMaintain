#ifndef SESSION_H
#define SESSION_H
#define BUF_SIZE 102400
#include <QFile>
#include <Qdir>
#include <QTcpSocket>
#include <sstream>
#include <QObject>
#include <QThread>
#include <QProgressBar>
#include <libssh/libssh2.h>
#include <libssh/libssh2_sftp.h>


class sessionBase{
public:
    QString host_ip;
    QString username;
    QString passwd;
    int port;
    QTcpSocket      *_socket;
    LIBSSH2_SESSION *_session;
    LIBSSH2_SFTP    *_sftp_session;
    sessionBase(QString host_ip, QString username,QString passwd,int port):
        host_ip(host_ip),username(username),passwd(passwd),port(port){
    }
    ~sessionBase();
    int init();
    int close();
};


class sessionIO: public QObject, public sessionBase
{
    Q_OBJECT
public:
    int _step = 3;
    explicit sessionIO(QString host_ip, QString username,QString passwd,int port, QObject *parent = nullptr);
    void download(QString local_file, QString remote_file);
    void initSessionIO();
signals:
    void process(int x);
    void finished(int x);
};


class session: public QObject, public sessionBase
{
    Q_OBJECT
public:
    explicit session(QString host_ip, QString username,QString passwd,int port, QObject *parent = nullptr);
    ~session();
    int _downloadFiles(QString file, QString remote_source_file);
    int uploadFileViaSftp(const QString &local_filename, const QString &remote_filename);
    QThread         *pThread;
    sessionIO       *_sessio;
    LIBSSH2_CHANNEL *_channel;
    long long _finished=0;
    long long _total=0;
    char sshBuf[BUF_SIZE];
    char testFile(QString file);
    void uploadFiles(QString file, QString remote_file);
    void downloadFiles(QString file, QString remote_source_file);
    QString execSSH(QString command);
    void process(int x);
    int close();
signals:
    void initSessionIO();
    void download(QString local_file, QString remote_file);
    void updateBar(int x);
};


#endif // SESSION_H

















