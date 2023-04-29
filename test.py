from datetime import datetime

datetime_str = 'December 3, 2022 at 1:53 PM EST'[0:-4]
# date_obj = datetime.strptime(date_str, '%B %d, %Y at %I:%M %p %Z')
datetime_object = datetime.strptime(datetime_str, '%B %d, %Y at %I:%M %p') # '%B %d, %Y at %I:%M %p %Z'

print(datetime_object)  # printed in default format
#########################################

n1 = 3884
n2 = 4921

def create_xticks(n1, n2):
  n1 = int(n1 / 10)
  n2 = int(n2 / 10)

  arr = []
  for x in range (n1 * 10, n2 * 10, 25):
    arr.append(x)

  return arr

# print(create_xticks(n1,n2))
