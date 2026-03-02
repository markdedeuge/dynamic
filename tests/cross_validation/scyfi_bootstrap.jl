# Bootstrap for loading SCYFI source files in a flat module.
# Skips `using .. Utilities` from scyfi_algo.jl since helpers are
# already included directly. Also skips `include("scyfi_algo.jl")`
# from find_cycles.jl since we include it ourselves.

# Read scyfi_algo.jl, skip the `using .. Utilities` line
let path = joinpath(@__DIR__, "..", "..", "reference", "SCYFI", "src", "scyfi_algo", "scyfi_algo.jl")
    code = read(path, String)
    # Remove the `using .. Utilities` line
    code = replace(code, r"using\s+\.\.\s*Utilities\s*\r?\n" => "")
    include_string(@__MODULE__, code, path)
end

# Read find_cycles.jl, skip the `include("scyfi_algo.jl")` line
let path = joinpath(@__DIR__, "..", "..", "reference", "SCYFI", "src", "scyfi_algo", "find_cycles.jl")
    code = read(path, String)
    # Remove the include of scyfi_algo.jl (we already loaded it above)
    code = replace(code, r"include\(\"scyfi_algo\.jl\"\)\s*\r?\n" => "")
    include_string(@__MODULE__, code, path)
end
