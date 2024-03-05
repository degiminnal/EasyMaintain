#ifndef ULTS_H
#define ULTS_H

#include <QtCore>
#include <QtGui>
#include <QDesktopWidget>
#include <bitset>

class Ults: public QObject
{
public:
    static QString ConvertMStoHMSMS(unsigned int ms)
    {
        uint32_t uHour;
        uint32_t uMin;
        uint32_t uSec;
        uint32_t uMis;
        uMis = ms % 1000;
        uSec = (uint32_t)(ms/1000) % 60;
        uMin = (uint32_t)(ms/60000) % 60;
        uHour = (uint32_t)(ms/3600000);
        QString timeStr = QString(tr("%1:%2:%3"))
                          .arg(uHour, 2, 10, QLatin1Char('0'))
                          .arg(uMin, 2, 10, QLatin1Char('0'))
                          .arg(uSec, 2, 10, QLatin1Char('0'));
        return timeStr;
    }

    static uint8_t SetBit(uint8_t oriValue, uint8_t startBit, uint8_t bitCount, uint8_t newValue)
    {
        std::bitset<8> b1(oriValue);
        std::bitset<8> b2(newValue);
        for (int i = 0; i < bitCount; i++)
        {
            b1[i+startBit] = b2[i];
        }

        uint8_t rv = b1.to_ulong() & 0xFF;
        return rv;
    }
};


#endif // ULTS_H
