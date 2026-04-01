import re

file_path = r'c:\Users\Vincenzo\Desktop\METAHACKATHON\env\env.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add _is_done to step()
target_step = r"""        truncated: bool = (
            not terminated
            and self._episode_step >= self._task.max_steps
        )"""

replacement_step = r"""        truncated: bool = (
            not terminated
            and self._episode_step >= self._task.max_steps
        )
        self._is_done = terminated or truncated"""

content = content.replace(target_step, replacement_step)

# Fix 2: Add deep_cast_numpy to state()
target_state = r"""        task_info: dict[str, Any] = self._task.get_info() if self._task else {}"""

replacement_state = r"""        def _deep_cast_numpy(obj):
            import numpy as np
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return [_deep_cast_numpy(i) for i in obj]
            if isinstance(obj, dict): return {str(k): _deep_cast_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_deep_cast_numpy(i) for i in obj]
            return obj
            
        task_info: dict[str, Any] = _deep_cast_numpy(self._task.get_info()) if self._task else {}"""

content = content.replace(target_state, replacement_state)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied")
