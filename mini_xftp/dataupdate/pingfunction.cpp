#include "pingfunction.h"

PingFunction::PingFunction(QWidget *parent) : QWidget(parent)
{

}

PingFunction::~PingFunction()
{
    qDebug()<<"析构函数"<<endl;
}
