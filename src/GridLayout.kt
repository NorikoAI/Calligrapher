@file:JsModule("@egjs/react-infinitegrid")
@file:JsNonModule
package react.infinitegrid

import react.RClass
import react.RProps
import kotlin.js.Json


external interface GridLayoutProps: RProps {
    var tag: String
    var containerTag: String
    var align: String
    var size: Int
    var margin: Int
    var isEqualSize: Boolean
    var outline: Array<*>
    var threshold: Int
    var isOverflowScroll: Boolean
    var isConstantSize: Boolean
    var useFit: Boolean
    var useRecycle: Boolean
    var useFirstRender: Boolean
    var horizontal: Boolean
    var percentage: Boolean
    var transitionDuration: Number
    var options: Json
    var layoutOptions: Json
    var layoutType: dynamic //Class
    var status: dynamic //IInfiniteGridStatus
    var loading: dynamic //React.ReactElement loading={<div>Loading...</div>}
    var onAppend: (dynamic)->Unit
    var onPrepend: (dynamic)->Unit
    var onLayoutComplete: (dynamic)->Unit
    var onImageError: (dynamic)->Unit
    var onChange: (dynamic)->Unit
    var items: dynamic
    var itemBy: (dynamic)->Int
    var groupBy: (dynamic)->Int
}

@JsName("GridLayout")
external val GridLayout : RClass<GridLayoutProps>