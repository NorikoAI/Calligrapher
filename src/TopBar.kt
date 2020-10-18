package org.sourcekey.NorikoAI.Calligrapher

import MaterialUI.core.*
import MaterialUI.icons.MenuIcon
import MaterialUI.icons.SearchIcon
import kotlinx.browser.window
import kotlinx.css.*
import kotlinx.html.js.onClickFunction
import react.*
import react.dom.button
import react.dom.div
import styled.css
import styled.styledDiv


interface TopBarProps : RProps {

}

interface TopBarState : RState {

}

private val topBar = functionalComponent<TopBarProps>{
    val (isMenuShow, setMenuShow) = useState(false)
    AppBar{
        attrs{
            color = "secondary"
            position = "static"
        }
        Toolbar{
            attrs.variant = "dense"
            IconButton{
                attrs{
                    edge = "start"
                    color = "inherit"
                    asDynamic().onClick = fun(){
                        if(isMenuShow){setMenuShow(false)}
                        else{setMenuShow(true)}
                    }
                }
                MenuIcon{}
            }
            Typography{
                attrs{
                    variant = "h6"
                }
                +"書法家"
            }
            styledDiv {
                css {
                    position = Position.relative
                    borderRadius = 0.3.vh
                    marginLeft = 3.vh
                    width = LinearDimension.auto
                    backgroundColor = Color( "rgba(0,0,0,0.1)")
                    hover {
                        width = 30.vh
                        backgroundColor = Color( "rgba(255,255,255,0.1)")
                    }
                }
                styledDiv {
                    css{
                        padding = "0.1vh"
                        height = LinearDimension("100%")
                        position = Position.absolute
                        pointerEvents = PointerEvents.none
                        display = Display.flex
                        alignItems = Align.center
                        justifyContent = JustifyContent.center
                    }
                    SearchIcon{}
                }
                InputBase{
                    attrs{
                        placeholder = "Search…"
                    }
                }
            }
        }
    }
    if(isMenuShow){
        styledDiv {
            css {
                background = "#FFF"
                width = 40.vh
                height = 100.vh
            }
            menu()
        }
    }
}

fun RBuilder.topBar(){
    child(topBar)
}