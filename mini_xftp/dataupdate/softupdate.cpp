#include "softupdate.h"
#include "ui_softupdate.h"

SoftUpdate::SoftUpdate(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SoftUpdate)
{
    ui->setupUi(this);
    setLED(ui->led1,0);
    setLED(ui->led2,0);
    setLED(ui->led3,0);
    setLED(ui->led4,0);
}

SoftUpdate::~SoftUpdate()
{
    delete ui;
}

void SoftUpdate::setLED(QLabel * label, int color)
{
    label->setText("");
    QString stylesheet = "min-width:16px;min-height:16px;max-width:16px;max-height:16px;border-radius:8px;border:1px solid black;";
    QString background = "background-color:";
    switch(color){
    case 0:
        background +="rgb(255,0,0)"; //红色
        break;
    case 1:
        background += "rgb(0,255,0)"; //绿色
        break;
    default:
        break;
    }
    const QString s = stylesheet+background;
    label->setStyleSheet(s);
}

void SoftUpdate::tabChange(int x)
{
    QString s = ui->textEdit->toPlainText();
    s += QString::asprintf("current tab: %d\n", x);
    ui->textEdit->setText(s);
}
