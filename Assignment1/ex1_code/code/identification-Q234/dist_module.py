# compute chi2 distance between x and y
def dist_chi2(x,y):
  a = x + y
  b = ((x - y)**2)
  x = a != 0 #as we dont want to divide with 0, so we check for index where vector value i not 0
  return sum(b[x]/a[x])

# compute l2 distance between x and y
def dist_l2(x,y):
  #Computing sum of squared distances
  return sum((x-y)**2)
 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similar histograms
def dist_intersect(x,y):
  return 1-sum([min(x[i],y[i]) for i in range(x.size)])

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name