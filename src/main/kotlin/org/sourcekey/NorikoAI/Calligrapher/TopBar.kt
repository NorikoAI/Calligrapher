package org.sourcekey.NorikoAI.Calligrapher

import MaterialUI.icons.*
import com.ccfraser.muirwik.components.*
import com.ccfraser.muirwik.components.button.*
import com.ccfraser.muirwik.components.input.*
import com.ccfraser.muirwik.components.menu.mMenu
import com.ccfraser.muirwik.components.menu.mMenuItemWithIcon
import kotlinx.css.*
import kotlinx.html.InputType
import kotlinx.html.js.onClickFunction
import kotlinx.html.style
import react.*
import react.dom.button
import react.dom.jsStyle
import styled.css
import styled.styledButton
import styled.styledDiv
import kotlin.js.RegExp


interface TopBarProps : RProps {
    var onSearchClick: (unicodeHex: String)->Unit
}

interface TopBarState : RState {

}

private val topBar = functionalComponent<TopBarProps>{props ->
    val (isMenuShow, setMenuShow) = useState(false)
    mAppBar{
        css {
            position = Position.static
        }
        mToolbar{
            attrs.variant = ToolbarVariant.dense
            mIconButton{
                attrs{
                    edge = MIconEdge.start
                    color = MColor.inherit
                    onClick = fun(e){
                        if(isMenuShow){setMenuShow(false)}
                        else{setMenuShow(true)}
                    }
                }
                MenuIcon{}
            }
            mTypography{
                attrs{
                    variant = MTypographyVariant.h6
                }
                +"書法家"
            }
            styledDiv {
                css {
                    position = Position.relative
                    borderRadius = 0.3.vh
                    marginLeft = 3.vh
                    backgroundColor = Color( "rgba(0,0,0,0.1)")
                    hover {
                        backgroundColor = Color( "rgba(255,255,255,0.1)")
                    }
                    focusWithin {
                        width = 30.vh
                        backgroundColor = Color( "rgba(255,255,255,0.1)")
                    }
                }
                styledDiv {
                    css { display = Display.flex }
                    styledDiv {
                        css {
                            width = 3.vh
                        }
                        SearchIcon{}
                    }
                    mInput(placeholder = "Unicode"){
                        attrs.onChange = fun(e){
                            var value = e.target.asDynamic().value as? String?:"0"
                            value = value.toIntOrNull(16)?.toString(16)?:return
                            props.onSearchClick(value)
                        }
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

fun RBuilder.topBar(onSearchClick: (String)->Unit){
    child(topBar){
        attrs.onSearchClick = onSearchClick
    }
}