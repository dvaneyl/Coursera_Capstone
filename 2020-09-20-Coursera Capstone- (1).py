#!/usr/bin/env python
# coding: utf-8

# In[1]:


# check library version numbers
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)


# In[2]:


import pandas as pd

df = pd.read_csv (r'C:\Users\Diderico van Eyl\OneDrive - DWNTRUST\Managing Center\3-Executing-Projects\00000-Coursera\Capstone-Course 9\2020-09-IBM-Coursera-Capstone.csv')


# In[3]:


import numpy as np


# In[4]:


df.head()


# In[5]:


df.shape[0]


# In[6]:


df.shape[1]


# In[12]:


df['SEVERITYCODE'].value_counts()


# In[ ]:


df.isnull().sum()


# In[13]:


df['SEVERITYCODE'].value_counts()


# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


plt.scatter(df.ST_COLCODE, df.SEVERITYCODE, color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[16]:


cdf = df[['SEVERITYCODE','ST_COLCODE']]
cdf.head(9)


# In[17]:


plt.scatter(cdf.ST_COLCODE, cdf.SEVERITYCODE,  color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[18]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[19]:


plt.scatter(train.ST_COLCODE, train.SEVERITYCODE,  color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[20]:


cdf2 = cdf[(cdf.SEVERITYCODE==2) & (cdf.ST_COLCODE==10)]


# In[21]:


cdf2.head()


# In[23]:


msk = np.random.rand(len(cdf2)) < 0.8
train = cdf2[msk]
test = cdf2[~msk]


# In[24]:


plt.scatter(train.ST_COLCODE, train.SEVERITYCODE,  color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[25]:


msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[26]:


plt.scatter(train.ST_COLCODE, train.SEVERITYCODE,  color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[27]:


cdf.head(10)


# In[28]:


df.head()


# In[31]:


plt.scatter(df.INCDATE, df.SEVERITYCODE==2,color='blue')
plt.xlabel("INCDATE")
plt.ylabel("SEVERITYCODE==2")
plt.show()


# In[145]:


plt.scatter(df.ST_, df.SEVERITYCODE, color='blue')
plt.xlabel("ST_COLCODE")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[11]:


import pandas as pd
dfml = pd.read_csv (r'C:\Users\Diderico van Eyl\OneDrive - DWNTRUST\Managing Center\3-Executing-Projects\00000-Coursera\Capstone-Course 9\DataCollisions-TrainingSet.csv')


# In[12]:


dfml.head()


# In[14]:


import numpy as np
msk = np.random.rand(len(dfml)) < 0.8
train = dfml[msk]
test = dfml[~msk]


# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(train.IncidentMonth, train.SEVERITYCODE,  color='blue')
plt.xlabel("IncidentMonth")
plt.ylabel("SEVERITYCODE")
plt.show()


# In[20]:


plt.scatter(train.IncidentWeekDay, train.SEVERITYCODE,  color='blue')
plt.xlabel(("IncidentWeekDay"))
plt.ylabel("SEVERITYCODE")
plt.show()


# In[ ]:




