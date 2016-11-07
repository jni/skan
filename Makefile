numba-clean:
	 find . -name "*.nbc" -print0 | xargs -0 rm
	 find . -name "*.nbi" -print0 | xargs -0 rm
