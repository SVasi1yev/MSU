/****************************************************************************
** Meta object code from reading C++ file 'PD_widget.hxx'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/pdview/PD_widget.hxx"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PD_widget.hxx' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_PD_widget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      18,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      15,   11,   10,   10, 0x05,
      46,   10,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
      70,   65,   10,   10, 0x0a,
      86,   10,   10,   10, 0x2a,
      98,   65,   10,   10, 0x0a,
     114,   10,   10,   10, 0x2a,
     130,  126,   10,   10, 0x0a,
     144,  126,   10,   10, 0x0a,
     158,  126,   10,   10, 0x0a,
     181,  172,   10,   10, 0x0a,
     203,  126,   10,   10, 0x2a,
     220,  172,   10,   10, 0x0a,
     242,  126,   10,   10, 0x2a,
     259,  126,   10,   10, 0x0a,
     275,   10,   10,   10, 0x0a,
     290,  126,   10,   10, 0x0a,
     314,  172,   10,   10, 0x0a,
     340,  126,   10,   10, 0x2a,

       0        // eod
};

static const char qt_meta_stringdata_PD_widget[] = {
    "PD_widget\0\0x,y\0mousePosChanged(double,double)\0"
    "uncheckPlotRatio()\0updt\0logX_slot(bool)\0"
    "logX_slot()\0logY_slot(bool)\0logY_slot()\0"
    "val\0P0_slot(bool)\0P1_slot(bool)\0"
    "P2_slot(bool)\0val,updt\0showH_slot(bool,bool)\0"
    "showH_slot(bool)\0hideP_slot(bool,bool)\0"
    "hideP_slot(bool)\0logH_slot(bool)\0"
    "recompH_slot()\0setMouseMode_slot(bool)\0"
    "plotRatio_slot(bool,bool)\0"
    "plotRatio_slot(bool)\0"
};

void PD_widget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        PD_widget *_t = static_cast<PD_widget *>(_o);
        switch (_id) {
        case 0: _t->mousePosChanged((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 1: _t->uncheckPlotRatio(); break;
        case 2: _t->logX_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->logX_slot(); break;
        case 4: _t->logY_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->logY_slot(); break;
        case 6: _t->P0_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->P1_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->P2_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->showH_slot((*reinterpret_cast< bool(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 10: _t->showH_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 11: _t->hideP_slot((*reinterpret_cast< bool(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 12: _t->hideP_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 13: _t->logH_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 14: _t->recompH_slot(); break;
        case 15: _t->setMouseMode_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 16: _t->plotRatio_slot((*reinterpret_cast< bool(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 17: _t->plotRatio_slot((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData PD_widget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject PD_widget::staticMetaObject = {
    { &QMathGL::staticMetaObject, qt_meta_stringdata_PD_widget,
      qt_meta_data_PD_widget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &PD_widget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *PD_widget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *PD_widget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_PD_widget))
        return static_cast<void*>(const_cast< PD_widget*>(this));
    return QMathGL::qt_metacast(_clname);
}

int PD_widget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMathGL::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    return _id;
}

// SIGNAL 0
void PD_widget::mousePosChanged(double _t1, double _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void PD_widget::uncheckPlotRatio()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}
QT_END_MOC_NAMESPACE
