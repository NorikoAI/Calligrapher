@file:JsModule("@egjs/react-infinitegrid")
@file:JsNonModule
package react.infinitegrid

import react.RClass
import react.RProps



external interface GridLayoutProps: RProps {
    var options: dynamic
    var layoutOptions: dynamic
    var onAppend: (dynamic)->Unit
    var onLayoutComplete: (dynamic)->Unit
}

@JsName("GridLayout")
external val GridLayout : RClass<GridLayoutProps>