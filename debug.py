
def trace(fn):
	def wrapper(*args, **kargs):
		print("Enter {}".format(fn.__name__))
		ret = fn(*args, **kargs)
		print("Leave {}".format(fn.__name__))
		return ret
	return wrapper

