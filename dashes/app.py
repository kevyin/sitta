# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


import pandas as pd

df = pd.read_csv(
        'https://gist.githubusercontent.com/chriddyp/'
        'c78bf172206ce24f77d6363a2d754b59/raw/'
        'c353e8ef842413cae56ae3920b8fd78468aa4cb2/'
        'usa-agricultural-exports-2011.csv')

df_graph = pd.read_csv(
        'https://gist.githubusercontent.com/chriddyp/' +
        '5d1ea79569ed194d432e56108a04d188/raw/' +
        'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
        'gdp-life-exp-2007.csv')

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
* Hello Hello
* Hello, it's me

I was wondering if after all these years you'd like to meet

To go over everything

They say that time's supposed to heal ya

But I ain't done much healing
Hello, can you hear me?
I'm in California dreaming about who we used to be
When we were younger and free
I've forgotten how it felt before the world fell at our feet
There's such a difference between us
And a million miles
Hello from the other side
I must have called a thousand times
To tell you I'm sorry for everything that I've done
But when I call you never seem to be home
Hello from the outside
At least I can say that I've tried
To tell you I'm sorry for breaking your heart
But it don't matter it clearly doesn't tear you apart anymore
Hello, how are you?
'''

def gen_graph(df):
    return dcc.Graph(
            id='life-exp-vs-gdp',
            figure={
                'data': [
                    go.Scatter(
                            x=df[df['continent'] == i]['gdp per capita'],
                            y=df[df['continent'] == i]['life expectancy'],
                            text=df[df['continent'] == i]['country'],
                            mode='markers',
                            opacity=0.7,
                            marker={
                                'size': 15,
                                'line': {'width': 0.5, 'color': 'white'}
                            },
                            name=i
                    ) for i in df.continent.unique()
                    ],
                'layout': go.Layout(
                        xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                        yaxis={'title': 'Life Expectancy'},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 0, 'y': 1},
                        hovermode='closest'
                )
            }
    )


def generate_table(dataframe, max_rows=10):
    return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                         html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                         ]) for i in range(min(len(dataframe), max_rows))]
    )

app = dash.Dash()

colors = {
    # 'background': '#111111',
    'background': '#white',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
            children='Hello Dash',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
            id='example-graph-2',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    }
                }
            }
    ),
    html.H4(children='US Agriculture Exports (2011)'),
    generate_table(df),
    html.H4(children='Life Exp vs GDP'),
    gen_graph(df_graph),
    dcc.Markdown(children=markdown_text)

])




if __name__ == '__main__':
    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
    app.run_server(debug=True)
