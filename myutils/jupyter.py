from IPython import get_ipython
from IPython.core.magic import register_cell_magic

__all__ = [
    'log_errors',
]


@register_cell_magic
def log_errors(line, cell):
    """Log errors in cell via the given logger.
    
    To use, put the following at start of cell:
        %%log_errors logger
    """
    
    logger_var_name = line or 'logger'
    eval(f'global {logger_var_name}')
    eval(logger_var_name)

    lines = cell.splitlines()
    tab = ' ' * 4
    
    new_cell = "try:\n"
    
    for line in cell.splitlines():
        new_cell += '    ' + line + "\n"
        
    new_cell += (
        f"except Exception as exc:\n"
        f"    {logger_var_name}.exception('error while executing cell')\n"
        f"    raise exc"
    )
    
    get_ipython().run_cell(new_cell)
