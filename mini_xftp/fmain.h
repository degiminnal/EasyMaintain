#ifndef FMAIN_H
#define FMAIN_H


#include <QMainWindow>
#include <QGroupBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QMessageBox>
#include <QLineEdit>
#include <QList>
#include <QVector>
#include <QListWidget>
#include <QHBoxLayout>
#include <QDebug>
#include <QByteArray>
#include <QFile>
#include <QTextStream>
#include <QFileDialog>

#include "datadown.h"
#include "dataupdate/softupdate.h"

namespace Ui {
class fmain;
}

class fmain : public QWidget
{
    Q_OBJECT

public:
    explicit fmain(QWidget *parent = nullptr);
    ~fmain();

private slots:
    void on_pTabWidget_currentChanged(int index);

    void on_tabWidget_currentChanged(int index);

private:
    Ui::fmain *ui;
    DataDown * datadown;
   // DataUpdate * dataupdate;
    SoftUpdate * softupdate;
};

#endif // FMAIN_H
