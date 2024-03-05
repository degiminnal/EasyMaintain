#include "fmain.h"
#include "ui_fmain.h"
#include <QDebug>


fmain::fmain(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::fmain)
{
    ui->setupUi(this);

    resize(800,800);
    QHBoxLayout * lay = new QHBoxLayout(this);
    QTabWidget * pTabWidget = new QTabWidget(this);
    pTabWidget->setTabsClosable(true);
    pTabWidget->setMovable(true);
    pTabWidget->setTabPosition(QTabWidget::North);
    pTabWidget->setTabShape(QTabWidget::Rounded);

    pTabWidget->setWindowTitle("数据分析小工具");
    datadown = new DataDown();
   // dataupdate = new DataUpdate();
    softupdate = new SoftUpdate();

    pTabWidget->insertTab(0,datadown, "数据下载");
   // pTabWidget->insertTab(2,dataupdate,"软件安装");
    pTabWidget->insertTab(1,softupdate,"软件安装");
    connect(pTabWidget,&QTabWidget::currentChanged,this,&on_pTabWidget_currentChanged);
    connect(pTabWidget,&QTabWidget::currentChanged,softupdate,&SoftUpdate::tabChange);
    lay->addWidget(pTabWidget);
}

fmain::~fmain()
{
    delete ui;
}

void fmain::on_pTabWidget_currentChanged(int index)
{
    qDebug()<<"on_tabWidget_currentChanged:"<<index<<endl;
}

void fmain::on_tabWidget_currentChanged(int index)
{

}
