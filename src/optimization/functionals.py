# In functionals.py:
import numpy as np

class LogScaledFunctional:
    def __init__(self, rf, optimizer, log_params):
        self.rf = rf
        self.opt = optimizer
        self.log_params = log_params
        
        # Proxy attributes that pyadjoint's minimize expects
        self.controls = optimizer.controls
        
    def __call__(self, controls):
        # Convert from log to physical space
        physical_controls = []
        for i, (name, ctrl) in enumerate(zip(self.opt.control_names, controls)):
            if name in self.log_params:
                # Input is in log space, convert to physical
                log_val = float(ctrl.dat.data[0]) if hasattr(ctrl, 'dat') else float(ctrl)
                phys_func = ctrl.copy(deepcopy=True) if hasattr(ctrl, 'copy') else self.opt.control_functions[i].copy(deepcopy=True)
                phys_func.dat.data[:] = 10**log_val
                physical_controls.append(phys_func)
            else:
                physical_controls.append(ctrl)
        
        return self.rf(physical_controls)
    
    def derivative(self):
        # Get gradient in physical space
        phys_grad = self.rf.derivative()
        
        # Convert to log space gradient
        log_grads = []
        for i, (name, grad) in enumerate(zip(self.opt.control_names, phys_grad)):
            if name in self.log_params:
                # Chain rule: d/d(log10(x)) = x * ln(10) * d/dx
                current_val = float(self.opt.control_functions[i].dat.data[0])
                grad_copy = grad.copy(deepcopy=True)
                grad_copy.dat.data[:] *= current_val * np.log(10)
                log_grads.append(grad_copy)
            else:
                log_grads.append(grad)
        
        return log_grads
    
    # Proxy methods that pyadjoint needs
    def obj_to_array(self, obj):
        """Convert controls to array - proxy to underlying RF"""
        return self.rf.obj_to_array(obj)
    
    def array_to_obj(self, array):
        """Convert array to controls - proxy to underlying RF"""
        return self.rf.array_to_obj(array)
    
    def __getattr__(self, name):
        """Proxy any other attributes to the underlying ReducedFunctional"""
        return getattr(self.rf, name)