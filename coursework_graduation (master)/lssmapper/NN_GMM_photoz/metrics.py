import numpy as np

# Функции для расчета нормализованного медианного нормального отклонения - 
# общая и с нижним порогом по Z

def sigma_nmad(true, preds):
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))

def sigma_nmad_z3(true, preds):
    mask = (true >= 3)
    true = true[mask]
    preds = preds[mask]
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))

def sigma_nmad_z4(true, preds):
    mask = (true >= 4)
    true = true[mask]
    preds = preds[mask]
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))

def sigma_nmad_z5(true, preds):
    mask = (true >= 5)
    true = true[mask]
    preds = preds[mask]
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))

def sigma_nmad_z6(true, preds):
    mask = (true >= 6)
    true = true[mask]
    preds = preds[mask]
    diff = preds - true
    m = np.median(diff)
    return 1.48 * np.median(np.abs((diff - m) / (1 + true)))

# Функции для доли катастрофических выбросов - 
# общая и с нижним порогом по Z

def out_rate(true, preds):
    diff = preds - true
    return ((np.abs(diff) / (1 + true)) > 0.15).sum() / len(true)

def out_rate_z3(true, preds):
    m = (true >= 3)
    true = true[m]
    preds = preds[m]
    diff = preds - true
    return ((np.abs(diff) / (1 + true)) > 0.15).sum() / len(true)

def out_rate_z4(true, preds):
    m = (true >= 4)
    true = true[m]
    preds = preds[m]
    diff = preds - true
    return ((np.abs(diff) / (1 + true)) > 0.15).sum() / len(true)

def out_rate_z5(true, preds):
    m = (true >= 5)
    true = true[m]
    preds = preds[m]
    diff = preds - true
    return ((np.abs(diff) / (1 + true)) > 0.15).sum() / len(true)

def out_rate_z6(true, preds):
    m = (true >= 6)
    true = true[m]
    preds = preds[m]
    diff = preds - true
    return ((np.abs(diff) / (1 + true)) > 0.15).sum() / len(true)