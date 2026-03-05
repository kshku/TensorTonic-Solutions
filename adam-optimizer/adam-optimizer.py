import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m)
    v = np.array(v)
    param = np.array(param)
    grad = np.array(grad)
    
    mt = (beta1 * m) + (1 - beta1) * grad
    vt = (beta2 * v) + (1 - beta2) * (grad ** 2)
    mcap = mt / (1 - (beta1 ** t))
    vcap = vt / (1 - (beta2 ** t))
    paramt = param - (lr * (mcap / (np.sqrt(vcap) + eps)))
    return (paramt, mt, vt)
    