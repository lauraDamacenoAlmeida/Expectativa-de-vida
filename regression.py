import streamlit as st
import pandas as pd
import base64
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def main():
    st.title('APS Regressão Linear ')
    st.text('Gabriel Oliveira Ramos do Nascimento RA: 21022939 \nJackson do Nascimento Silva RA: 21022770 \nLaura  Damaceno de Almeida  RA: 20964736 \nVictor Hugo Kawabata Fuzaro RA: 20760102')
    st.image('image.png',width= 900)
    file = st.file_uploader('Escolha seu arquivo', type='csv')

    if file is not None:
        slider = st.slider('Quantidade de linhas',0,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('**Nome das colunas**')
        st.write(list(df.columns))
        st.markdown('**Número de linhas**')
        st.write(df.shape[0])
        st.markdown('**Número de colunas**')
        st.write(df.shape[1])
        exploracao = pd.DataFrame({'nomes': df.columns,'tipos':df.dtypes,'NA #': df.isna().sum(), 'NA %': df.isna().sum()/df.shape[0]*100})
        st.markdown('**Contagem dos tipos de dados**')
        st.write(exploracao.tipos.value_counts())
        st.markdown('**Nome das colunas do tipo int64**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']))
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))
        st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
        st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])
        st.markdown('**Descrição dos dados :**')

        st.table(df.describe())
        opcoes = df.columns
        selected_atributos = st.multiselect('Selecione os atributos', opcoes)
        type_visualize = st.selectbox('Selecione o tipo de visualização',['selecione','boxplot','scatter plot','barchart','histograma','Matriz de correlação'])
        df.dropna(inplace=True)
        if(len(selected_atributos)>2):
            st.markdown('Selecione no máximo 2 atributos')
        if (len(selected_atributos) <= 2):
            if(type_visualize == 'barchart'):
                plot_data = df[selected_atributos[0]]
                st.bar_chart(plot_data)
            if (type_visualize == 'boxplot'):
                if(len(selected_atributos)==1):
                    fig = px.box(df, y=selected_atributos[0],hover_data=['Country'])
                    #df.boxplot([selected_atributos[0]])
                else:
                    fig = px.box(df, x =selected_atributos[0], y=selected_atributos[1],hover_data=['Country'])
                    #df.boxplot([selected_atributos[0]], by=[selected_atributos[1]])
                #st.pyplot()
                st.plotly_chart(fig)
            if(type_visualize=='scatter plot'):
                if(len(selected_atributos)==1):
                    fig = px.scatter(df, x=selected_atributos[0], hover_data=['Country'])
                    st.plotly_chart(fig, use_container_width=True)

                if(len(selected_atributos)==2):
                    fig = px.scatter(df, x=selected_atributos[0], y=selected_atributos[1], hover_data=['Country'])
                    st.plotly_chart(fig, use_container_width=True)
            if (type_visualize == 'histograma'):
                sns.distplot(df[selected_atributos[0]])
                st.pyplot()
            if (type_visualize == 'Matriz de correlação'):
                st.write(df.corr())

        st.markdown('**Regressão Linear**')
        Y = st.selectbox('Selecione a variável Y', opcoes)
        x = st.multiselect('Selecione a variável X', opcoes)
        if((Y !=None)&(len(x)>=1)):
            modelo = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(df[x], df[Y], test_size=0.3, random_state=2811)
            modelo.fit(X_train, y_train)
            st.text("R quadrado = {}".format(modelo.score(X_train,y_train).round(2)))
            y_predict_train = modelo.predict(X_train)
            lm = modelo.predict(X_test)
            st.text("R quadrado de teste = {}".format(metrics.r2_score(y_test,lm).round(2)))
            sns.regplot(x = y_predict_train,y=y_train)
            st.pyplot()
            index = x
            index.append('Intercept')
            st.markdown('**Formula da Regressão Linear**')
            st.image('formula.png', width=500)
            st.table(pd.DataFrame(data=np.append(modelo.intercept_, modelo.coef_), index=index, columns=['Parametros']))


if __name__ == '__main__':
    main()
