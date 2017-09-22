n = input()
s=raw_input()

r=map(str,str(s))

#removing comments
for i in range(len(r)):
    if(i+1<len(r)):
        if(r[i]=="@" and r[i+1]=="*"):
           p=i
        if(r[i]=="*" and r[i+1]=="@"):
           q=i

del r[p:q]

a=0
b=0
c=0
d=0
e=0
for word in r:
    if(word=="("):
        a=a+1
    if(word==")"):
        a=a-1
    if(word=="["):
        b=b+1
    if(word=="]"):
        b=b-1
    if(word=="<"):
        c=c+1
    if(word==">"):
        c=c-1
    if(word=="{"):
        d=d+1
    if(word=="}"):
        d=d-1
    if(word=="/"):
        e=e+1

if (a==0 and b==0 and c==0 and d==0 and (e%2)==0 ):
    print "True"
else:
    print "False"
  
        
            
        
    