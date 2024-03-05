#ifndef SIGLETON_H
#define SIGLETON_H

#include <QtCore/QMutex>

#define DECL_SINGLETON(classname)   \
private:                            \
classname(){};                      \
~classname(){};                     \
static   classname * m_instance;    \
static QMutex  m_mutex;             \
public:                             \
static classname * instance();      \
static void freeInstance();

#define DECL_SINGLETON_ON_CONSTRUCTION(classname)   \
private:                            \
~classname(){};                     \
static   classname * m_instance;    \
static QMutex  m_mutex;             \
public:                             \
static classname * instance();      \
static void freeInstance();

#define DECL_SINGLETON_EXT(classname,baseclassname) \
private:                            \
classname():baseclassname(){ init(); };      \
~classname(){};                     \
static   classname * m_instance;    \
static QMutex  m_mutex;             \
public:                             \
static classname * instance();      \
static void freeInstance();

#define IMPL_SINGLETON(classname)   \
classname * classname::m_instance = NULL; \
QMutex classname::m_mutex;          \
classname * classname::instance() { \
if(!m_instance)                     \
{                                   \
    QMutexLocker lock(&m_mutex);    \
    if(!m_instance)                 \
        m_instance = new classname; \
}                                   \
return m_instance;                  \
}                                   \
void classname::freeInstance() {    \
    QMutexLocker lock(&m_mutex);    \
    if(m_instance) {                \
        delete m_instance; m_instance = NULL;   \
    }   \
}

#endif // SIGLETON_H
