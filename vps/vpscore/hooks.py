import weakref
import torch

class HookManager:
    def __init__(self):
        self.handles = []
        self.layers = weakref.WeakSet()
        self.head_outputs = {}  # for multi-head exterior

    def attach(self, model):
        self.detach()
        for name, m in model.named_modules():
            if m.__class__.__name__ == "VPSLinear":
                self.layers.add(m)
                self.handles.append(m.register_forward_pre_hook(self._fwd_pre))
                self.handles.append(m.register_forward_hook(self._fwd_post))
                self.handles.append(m.register_full_backward_hook(self._bwd))
            # Capture per-head outputs if attention block
            if "attention" in name.lower() and hasattr(m, "num_heads"):
                self.handles.append(m.register_forward_hook(self._capture_heads))

    def detach(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()
        self.layers = weakref.WeakSet()
        self.head_outputs.clear()

    @staticmethod
    def _fwd_pre(module, inputs):
        x = inputs[0]
        module.last_x = x.detach()

    @staticmethod
    def _fwd_post(module, inputs, output):
        pass

    @staticmethod
    def _bwd(module, grad_input, grad_output):
        g = grad_output[0]
        module.last_grad_h = g.detach()

    def _capture_heads(self, module, inputs, output):
        """Capture per-head outputs for exterior products"""
        if hasattr(output, "shape") and len(output.shape) >= 3:
            self.head_outputs[id(module)] = output.detach()

    def clear_buffers(self):
        for m in list(self.layers):
            for name in ("last_x","last_h","last_grad_h"):
                if hasattr(m, name): setattr(m, name, None)
        self.head_outputs.clear()

    def get_head_outputs(self, module_id):
        return self.head_outputs.get(module_id, None)


