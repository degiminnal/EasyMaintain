#ifndef CUSTOMPROGRESSDIALOG_H
#define CUSTOMPROGRESSDIALOG_H

#include <QProgressDialog>
#include "add/head.h"
class CustomProgressDialog : public QProgressDialog
{
public:
    CustomProgressDialog() : QProgressDialog()
    {
        initialize();
    }
private:
    void initialize()
    {
        QFont font("ZYSong18030", 12);
        this->setFont(font);
        this->setWindowModality(Qt::WindowModal);
        this->setMinimumDuration(5);
        this->setWindowTitle("正在运行，请等待...");
        this->setCancelButtonText(tr("取消"));
        this->setRange(0, 100);
    }
};

#endif // CUSTOMPROGRESSDIALOG_H
