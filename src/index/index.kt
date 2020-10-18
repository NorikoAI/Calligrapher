package index


import MaterialUI.core.*
import MaterialUI.icons.MenuIcon
import MaterialUI.icons.SearchIcon
import app.*
import kotlinext.js.*
import react.dom.*
import kotlinx.browser.*
import kotlinx.css.*
import kotlinx.html.style
import react.*
import react.infinitegrid.GridLayout
import styled.css
import styled.styledDiv
import ticker.TickerProps
import ticker.TickerState




external interface ItemProps : RProps {
    var num: Int
}

fun RBuilder.item(num: Int): RClass<ItemProps> {
    val item = rFunction<ItemProps>("Item") { props ->
        div{
            attrs.jsStyle {
                background = "#EEE"
                width = "250px"
                //height = "300px"
                overflow = "hidden"
            }
            img{
                attrs.jsStyle {
                    width = "250px"
                }
                attrs.src = "https://naver.github.io/egjs-infinitegrid/assets/image/${(num % 59) + 1}.jpg"
            }
        }
    }
    item.invoke {  }
    return item
}






fun main(){
    requireAll(require.context("src", true, js("/\\.css$/")))
    render(document.getElementById("root")) {
        app()
    }
}



