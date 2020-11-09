package logo

import react.*
import react.dom.*
import kotlinext.js.*
import kotlinx.html.style

val reactLogo = "img/react.svg"
val kotlinLogo = "img/kotlin.svg"

fun RBuilder.logo(height: Int = 100) {
    div("Logo") {
        attrs.jsStyle.height = height
        img(alt = "React logo.logo", src = reactLogo, classes = "Logo-react") {}
        img(alt = "Kotlin logo.logo", src = kotlinLogo, classes = "Logo-kotlin") {}
    }
}
