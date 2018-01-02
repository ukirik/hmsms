import sys

class OffsetList(list):
    """
    This is a utility class that provides convenient access
    using an offset. Such that with offset = 1 it will start
    indices from 1 instead of 0. With offset = 0, it's a regular list.
    """

    def __init__(self, offset=0):
        self.offset = offset
        list.__init__(self)

    def __getitem__(self, i):
        if 0 <= i < self.offset:
            raise KeyError
        try:
            return list.__getitem__(self, i - self.offset)
        except IndexError:
            print("caught error... debug this!")

    def __setitem__(self, i, y):
        if 0 <= i < self.offset:
            raise KeyError
        list.__setitem__(self, i - self.offset, y)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        if self.offset != other.offset:
            return False

        for i in range(self.offset, self.offset + len(self)):
            if self[i] != other[i]:
                return False

        return True


def getNbrOfLines(files):
    import subprocess
    total = 0
    if isinstance(files, str):
        list_of_files = [files]
    else:
        list_of_files = files

    assert isinstance(list_of_files, list)

    for f in list_of_files:
        cmd = f"wc -l {f}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        nlines = result.strip().split()[0]
        total += int(nlines)
    return total


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


# File generator for handling multiple input files
def yield_open(filenames):
    for filename in filenames:
        with open(filename, 'r') as file:
            yield file