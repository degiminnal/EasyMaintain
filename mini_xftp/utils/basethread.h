#ifndef BASETHREAD_H
#define BASETHREAD_H

#include <QObject>

typedef enum
{
    STATUS_INIT,
    STATUS_STOP,
    STATUS_PAUSE,
    STATUS_RUNNING
}THREAD_STATUS;

class BaseThread : public QObject
{
    Q_OBJECT
public:
    BaseThread();
    ~BaseThread();
signals:
    void WorkSignal();
    void StopSignal();
public slots:
    // 工作过程
    virtual void run() = 0;
protected:
    THREAD_STATUS status; // 线程状态
protected:
    // 初始化
    virtual void init() = 0;
public:
    // 暂停处理
    void _pause() { status = STATUS_PAUSE; };
    // 继续处理
    void _continue() { status = STATUS_RUNNING; };
    // 停止线程
    void _stop() { status = STATUS_STOP; };
    // 线程运行状态
    THREAD_STATUS _status() { return status; };
};

#endif // BASETHREAD_H
