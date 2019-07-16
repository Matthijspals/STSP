
data_path <- "C:/Users/Matthijs/Desktop/CSV/"

library(readr)
library(ggplot2)
library(reshape2)

#first analyse the output of the  output ensemble

#fill in before running!
ms<-498 #simulation time in ms-1 (only 499 ms was saved to save space)
probes<-c(-42,-33,-25,-18,-12,-7,-3, 3,7,12,18,25,33,42) #orientation differences
runs<-96 #amount of runs-1


#column names
names<-list("t", "-42","-33","-25","-18","-12","-7","-3", "3","7","12","18","25","33","42")
#load all csv files in data frame
dat <- data.frame(t=array(0:ms))
for (probe in probes) {
  for (run in 0:runs){

    name <-paste("81_Diff_Theta_",probe,"_run_",run,".csv",sep = "")
    newcol<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
    dat <- cbind(dat,newcol)

  }
}
#calculate means for decision relevant period
datmean <- data.frame(t<-array(15:100))
for (probe in 0:13){
  datmean<-cbind(datmean,rowMeans(dat[(probe*(runs+1)+2):(probe*(runs+1)+runs+2)])[215:300])

}
colnames(datmean) <- names

#create ggplot of representation in output ensemble
datmelted <-melt(datmean,  id.vars = 't', variable.name = 'probe')
ggplot(datmelted, aes(t,value)) +
    geom_line(aes(colour = probe), size= 1.0) + 
    theme_classic()+
    guides(colour=guide_legend(ncol=2,title='Orientation difference \nbetween memory \nitem and probe')) +
    ylab('Output decision ensemble')+
    xlab('Time after onset probe (ms)')



#Extract a decision from the output ensemble

names<-c("-42°","-33°","-25°","-18°","-12°","-7°","-3°", "3°","7°","12°","18°","25°","33°","42°")
dfm <- data.frame(matrix(ncol = 14, nrow = 0))
colnames(dfm) <- names
Nm<-1 #number of model participants (runs divided by 96)
for (i in 0:(Nm-1)){
  runsplit<-(i*96)+1
  runsplitruns<-(i*96+96)
  datans <- c(rep(0,14))
  for (probe in 0:13){
    for(run in runsplit:runsplitruns){
      #if the surface under the curve 30 to 80 ms after presentation of the probe 
      #is positive indicate a clockwise response
      if (sum(dat[230:280,(probe*(runs+1)+run)])>0){
        datans[probe+1]<-(datans[probe+1]+1)
      }
    }
  }
  print(datans/96)
  dfm[nrow(dfm)+1,] <-datans/96

}


x<-c(-42,-33,-25,-18,-12,-7,-3, 3,7,12,18,25,33,42)
#plot performance of model
plot(x,colMeans(dfm))


#load data if human participants
data_path <- "C:/Users/Matthijs/Desktop/Data_Wolff/Data/"
library(readr)
library(ggplot2)
library(reshape2)


names<-c("-42","-33","-25","-18","-12","-7","-3", "3","7","12","18","25","33","42")
df <- data.frame(matrix(ncol = 14, nrow = 0))
colnames(df) <- names
N<-30
scoreperpart<-c(rep(0,N))
for (p in 1:N){
  name <-paste("results",p,".csvi",sep = "")
  temp<-read_csv(paste0(data_path,name),col_names=FALSE,col_types=cols(X1 = col_double()))
  score<-c(rep(0,14))
  trials<-c(rep(0,14))
  for (i in 1: nrow(temp)){
    for (j in 1: length(items)){
      if (temp$X4[i] == items[j]){
        score[j]<-score[j]+temp$X5[i]
        trials[j]<-(trials[j]+1)
      }
    }
  }
  add<-score/trials
  for (i in 1:7){
    add[i]<-(1-add[i])
  }
  #print(score/trials)
  df[nrow(df)+1,] <-add
}

#include CI (95% of the mean)
ci<-c(rep(0,14))
ymin<-c(rep(0,14))
ymax<-c(rep(0,14))
for (col in 1:14){
  ci <- qt(0.975,df=N-1)*sd(df[1:N,col])/sqrt(N)
  ymax[col]<-mean(df[1:N,col])+ci
  ymin[col]<-mean(df[1:N,col])-ci
}
dfpart<- data.frame(items,colMeans(df),ci,ymin,ymax)
names(dfpart)<-c("x","y","ci","ymin","ymax")

#create ggplot of human participants performance
plot_exp<-ggplot(dfpart, aes(x=x,y=y)) +
  geom_line(size=1.3,colour = "turquoise4") +
  geom_errorbar(aes(ymin=ymin, ymax=ymax), width=1) +
  geom_point() + 
  geom_line(aes(x=0), color = "darkgrey", linetype = "dashed",size=1.3)+
  theme_classic()+
  scale_x_continuous(breaks=items) +
  scale_y_continuous(breaks=seq(0,1,by=0.1),limits = c(0,1)) +
  theme(legend.position="none") +
  ggtitle("Participants")+
  ylab('Proportion of clockwise response')+
  xlab('Angular Difference (°) between \n memory item and probe')

#create similar plot for model 
ci<-c(rep(0,14))
ymin<-c(rep(0,14))
ymax<-c(rep(0,14))
for (col in 1:14){
  ci <- qt(0.975,df=N-1)*sd(dfm[1:Nm,col])/sqrt(Nm)
  ymax[col]<-mean(dfm[1:Nm,col])+ci
  ymin[col]<-mean(dfm[1:Nm,col])-ci
}
dfmod<- data.frame(items,colMeans(dfm),ci,ymin,ymax)
names(dfmod)<-c("x","y","ci","ymin","ymax")

plot_mod<-ggplot(dfmod, aes(x=x,y=y)) +
  geom_line(size=1.3,colour = "turquoise4") +
  geom_errorbar(aes(ymin=ymin, ymax=ymax), width=1) +
  geom_point() + 
  geom_line(aes(x=0), color = "darkgrey", linetype = "dashed",size=1.3)+
  #geom_smooth(se=FALSE, size=1.3) +
  theme_classic()+
  scale_x_continuous(breaks=items) +
  scale_y_continuous(breaks=seq(0,1,by=0.1), limits = c(0,1)) +
  theme(legend.position="none") +
  ggtitle("Model")+
  ylab('Proportion of clockwise response')+
  xlab('Angular Difference (°) between \n memory item and probe')


#make plot including the performance of both the model and the human participants
library(cowplot)

p<-plot_grid(plot_exp, plot_mod, labels = "AUTO",label_x = 0, label_y = 0, hjust = -0.5, vjust = -0.5 ) 
title <- ggdraw()# + draw_label("Behavioural Performance", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1)) # rel_heights values control title margins

