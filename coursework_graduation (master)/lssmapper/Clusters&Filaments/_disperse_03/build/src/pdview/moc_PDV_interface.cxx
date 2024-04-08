/****************************************************************************
** Meta object code from reading C++ file 'PDV_interface.hxx'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/pdview/PDV_interface.hxx"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PDV_interface.hxx' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_PDV_interface[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      21,   15,   14,   14, 0x08,
      46,   14,   14,   14, 0x08,
      59,   55,   14,   14, 0x08,
      94,   14,   14,   14, 0x08,
     113,   14,   14,   14, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_PDV_interface[] = {
    "PDV_interface\0\0event\0closeEvent(QCloseEvent*)\0"
    "onQuit()\0x,y\0updateMousePosLabel(double,double)\0"
    "uncheckPlotRatio()\0helpPopup()\0"
};

void PDV_interface::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        PDV_interface *_t = static_cast<PDV_interface *>(_o);
        switch (_id) {
        case 0: _t->closeEvent((*reinterpret_cast< QCloseEvent*(*)>(_a[1]))); break;
        case 1: _t->onQuit(); break;
        case 2: _t->updateMousePosLabel((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 3: _t->uncheckPlotRatio(); break;
        case 4: _t->helpPopup(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData PDV_interface::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject PDV_interface::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_PDV_interface,
      qt_meta_data_PDV_interface, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &PDV_interface::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *PDV_interface::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *PDV_interface::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_PDV_interface))
        return static_cast<void*>(const_cast< PDV_interface*>(this));
    return QWidget::qt_metacast(_clname);
}

int PDV_interface::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
