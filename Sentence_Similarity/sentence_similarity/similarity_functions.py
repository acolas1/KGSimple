from math import *
'''calculate euclidean distance'''
def euclidean(x,y):
  try:
    return round(sqrt(sum(pow(a-b,2) for a, b in zip(x, y))),3)
  except Exception as e:
    print(e)
    return None

'''calculate manhattan distance'''
def manhattan(x,y):
  try:
    return round(sum(abs(a-b) for a,b in zip(x,y)),3)
  except Exception as e:
    print(e)
    return None

'''calculate minkowski distance'''
def minkowski(x,y):
    try:
        var=sum(pow(abs(a-b),3) for a,b in zip(x, y))
        nth_root_value = 1/3
        nth_root_result=var ** nth_root_value
        return round(nth_root_result,3)
    except Exception as e:
        print(e)
        return None

def square_rooted(x):
   try:
        return round(sqrt(sum([a*a for a in x])),3)
   except Exception as e:
        print(e)
        return None

'''calculate cosine score'''  
def cosine(x,y):
   try:
        numerator = sum([a*b for a,b in zip(x,y)])
        denominator = square_rooted(x)*square_rooted(y)
        return round(numerator/denominator,3)
   except Exception as e:
        print(e)
        return None

