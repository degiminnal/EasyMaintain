#ifndef SOFTUPDATE_H
#define SOFTUPDATE_H

#include <QWidget>
#include <QLabel>
#include <QDebug>

namespace Ui {
class SoftUpdate;
}

class SoftUpdate : public QWidget
{
    Q_OBJECT

public:
    explicit SoftUpdate(QWidget *parent = nullptr);
    ~SoftUpdate();

    void setLED(QLabel * label, int color);

public slots:
    void tabChange(int x);

private:
    Ui::SoftUpdate *ui;
};

#endif // SOFTUPDATE_H
