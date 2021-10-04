### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 5dbcd3de-fe92-11eb-01f7-fbe7e0714a6a
begin
  using JSON
  using MLDataPattern
  using Random
  using NPZ
  using ArgParse
  using Crayons.Box
  using NPZ
end

# ╔═╡ 190194c2-aea0-4053-85ed-b5d5d97632fb
begin
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--pattern"
    arg_type = Int
    default = 1
    "--samples_per_aspect"
    arg_type = Int
    default = 3
    "--nb_neg_samples"
    arg_type = Int
    default = 3
    "--pretrained_weight"
    arg_type = String
    default = "albert-base-v2"
    "--seed"
    arg_type = Int
    default = 42
    "--use_active_learning"
    arg_type = String
    default = "no"
    "--active_learning_threshold"
    arg_type = Float64
    default = 100.0
    "--act_shuffle"
    arg_type = String
    default = "false"
    "--guided_sampling"
    arg_type = String
    default = "false"
  end
  args = parse_args(ARGS, s)
end

# ╔═╡ 79098e1e-5917-484c-9324-757548f1f699
function set_run_options(filename, outfilename;
                         pattern=1, dataset="dosentencepairs",
                         pretrained_weight="albert-base-v2")
  js = JSON.parse(read(filename, String))
  js["pattern"] = pattern
  js["dataset"] = dataset
  js["pretrained_weight"] = pretrained_weight
  open(outfilename, "w") do io
    JSON.print(io, js)
  end
end

# ╔═╡ d8903bd9-38d2-4b62-9e2c-f46e1ef7f196
asp2asp = Dict(
  "Activities" => "Activities", "Ambiance" => "Environment", "Amenities" => "Extra", "Bathroom" => "Bathroom", "Bed" => "Bed", "Breakfast" => "Breakfast", "Cleanliness" => "Clean", "Dinner" => "Dinner", "Drinks" => "Drinks", "Family-friendliness" => "Family", "Food" => "Food", "Hotel" => "Hotel", "Location" => "Location", "Maintenance" => "Maintenance", "Noise level" => "Noise", "Payment" => "Payment", "Public transport" => "Transport", "Reception" => "Reception", "Room" => "Room", "Safety" => "Safety", "Staff" => "Staff", "Value for money" => "Value", "View" => "View", "Wellness" => "Spa", "WiFi" => "Internet"
)

# ╔═╡ faad7cdd-2811-46c4-9387-c6934b4e1359
function write_data(filename, data, nb_neg_samples; seed=42)
  allaspects = values(asp2asp)
  data_to_write = []
  
  idx = 1
  for r in data
    d = Dict(
      "idx" => idx,
      "hypothesis" => r["sentence"],
      "premise" => r["label"],
      "label" => "Yes")
    push!(data_to_write, d)
    idx += 1
    # Write the negative samples of this sentence
    for aspect in shuffle(setdiff(allaspects, [r["label"]]))[1:nb_neg_samples]
      d = Dict(
	"idx" => idx,
	"hypothesis" => r["sentence"],
	"premise" => aspect,
	"label" => "No")
      idx += 1
      push!(data_to_write, d)
    end
  end
  
  shuffle!(MersenneTwister(seed), data_to_write)
  
  open(filename, "w") do IO
    for d in data_to_write
      println(IO, json(d))
    end
  end
end

# ╔═╡ ea852da8-2bee-4f54-86f2-fe1222d38361
function write_all_data(filename, data)
  allaspects = values(asp2asp)
  data_to_write = []
  
  idx = 1
  for r in data
    for aspect in allaspects
      d = Dict(
	"idx" => idx,
	"hypothesis" => r["sentence"],
	"premise" => aspect,
	"label" => aspect == r["label"] ? "Yes" : "No")
      push!(data_to_write, d)
      idx+=1
    end
  end
  
  open(filename, "w") do IO
    for d in data_to_write
      println(IO, json(d))
    end
  end
end

# ╔═╡ 354d0e38-ebf3-4069-921e-90588ff8e78e
function generate_dataset(infilename, outdir;samples_per_aspect=3, nb_neg_samples=3, seed=42,
                          activelearningmetric=:entropy, active_learning_threshold=nothing,
                          act_shuffle=nothing, guided_sampling=nothing)
  # Load data and pre-process
  js = JSON.parse(read(infilename, String))
  sentences, aspects, sentiments = js["sentences"], js["aspects"], js["sentiments"]
  aspects = [asp2asp[a] for a in aspects]
  clean_sentences = [replace(sent, "\n" => "") |> strip for sent in sentences]
  numdict(X) = Dict(x => i for (i,x) in enumerate(X|>unique|>sort))
  asp2num = numdict(aspects)
  sent2num = numdict(sentiments)

  # Bring sentence/label data in the correct form
  res = []
  for (i,(sentence, aspect)) in enumerate(zip(clean_sentences, aspects))
    push!(res, Dict("sentence" => sentence, "label" => aspect, "idx"=>i))
  end
  
  # If active learning data is available, sort the sentences by how unsure we are
  if activelearningmetric != :no && isfile("entropy_and_breaking_ties.npz")
    println("Using Active learning data")
    npz = npzread("entropy_and_breaking_ties.npz")
    metric = activelearningmetric == :entropy ? npz["entropies"] : npz["breaking_ties"]
    # Filter sentences with a metric value above threshold. The idea
    # here is that extremely uncertain sentences might just not be
    # very useful
    sentmet = filter(x->x[2]<active_learning_threshold, zip(sentences, metric)|>collect)
    sentences = first.(sentmet)
    metric = last.(sentmet)
    # Sort sentences according to uncertainty
    sentences = sentences[sortperm(metric, rev=true)]
    # Shuffle a certain amount of the most confusing sentences to bring in a bit more variability
    num_shuffle = guided_sampling ? 100 : samples_per_aspect*2
    act_shuffle && shuffle!(@view(sentences[1:num_shuffle]))
  else
    # Shuffle data randomly
    shuffle!(MersenneTwister(seed), res)
  end
  
  if guided_sampling
    # Select a fixed number of samples of each aspect for the training set
    train_res = []
    rest_res = []
    asp_count = Dict(asp => 0 for asp in values(asp2asp))
    for r in res
      if asp_count[r["label"]] < samples_per_aspect
        push!(train_res, r)
        asp_count[r["label"]] += 1
      else
        push!(rest_res, r)
      end
    end
  else # If we are not doing guided sampling, randomly select (this is more realistic in practice)
    train_res = res[1:samples_per_aspect]
    rest_res = res[samples_per_aspect+1:end]
  end
  # Split off dev and test set
  dev_res, test_res = splitobs(shuffleobs(rest_res, rng=MersenneTwister(seed)),
		               at=0.02)
  
  isdir(outdir) || mkdir(outdir)
  
  write_data(joinpath(outdir, "train.jsonl"), train_res, nb_neg_samples, seed=seed)
  write_data(joinpath(outdir, "dev.jsonl"), dev_res, 0, seed=seed)
  write_all_data(joinpath(outdir, "test.jsonl"), test_res[1:1000])
end

# ╔═╡ b4587bec-c0fd-474f-9eb7-1a795c203452
function evaluate(resultdir, testfilename)
  # Get results from model
  outputfilename = joinpath(resultdir, "test_logits.npy")
  logits = npzread(outputfilename)
  modelasepcts = argmax.(Iterators.partition(logits[:,1], 25))
  # Get correct results
  testjs = JSON.parse.(readlines(testfilename))
  getaspectnr(partition) = findfirst(x->x["label"]=="Yes", partition)
  trueaspects = getaspectnr.(Iterators.partition(testjs, 25))
  # Calculate accuracy
  sum(modelasepcts .== trueaspects)/length(modelasepcts)
end

function get_gpu_lock()
  while true
    for gpu in 1:3
      gpu_file = "/home/sebastianstabinger/tmp/gpu_$(gpu).lock"
      if !isfile(gpu_file)
        touch(gpu_file)
        ENV["CUDA_VISIBLE_DEVICES"] = string(gpu)
        println(GREEN_FG("Using GPU $gpu"))
        Base.atexit() do
          unlock_gpu(gpu)
        end
        return gpu
      end
    end
    sleep(1)
  end
end

function unlock_gpu(gpu)
  filename = "/home/sebastianstabinger/tmp/gpu_$(gpu).lock"
  isfile(filename) && rm(filename)
end

# ╔═╡ 8e5fddff-33e8-4854-91d6-c8ee6895b3c5
function run_experiment(;pattern=1, samples_per_aspect=3, nb_neg_samples=3, 
		        pretrained_weight="albert-base-v2", seed=42,
                        activelearningmetric=:no, active_learning_threshold=nothing,
                        act_shuffle=nothing, guided_sampling=nothing)
  @assert activelearningmetric ∈ [:no, :entropy, :breaking_ties] "Active learning method unknown"
  
  ENV["PET_ELECTRA_ROOT"] = pwd()
  println("Trying to get GPU lock")
  gpu = get_gpu_lock() # Get a GPU lock
  println("running in directory $(pwd())")
  dataset_postfix = "$(pattern)_$(samples_per_aspect)_$(nb_neg_samples)_$(pretrained_weight)"
  datasetname = "dosentencepairs_$(dataset_postfix)"
  configfilename = joinpath("./config", datasetname*".json")
  # Generate datset
  println("Generating dataset")
  generate_dataset("./hotels_topic.json", joinpath("./data", datasetname), seed=seed,
                   samples_per_aspect=samples_per_aspect, activelearningmetric=activelearningmetric,
                   active_learning_threshold=active_learning_threshold, act_shuffle=act_shuffle,
                   guided_sampling=guided_sampling)
  # Generate config
  println("Setting run options")
  set_run_options("./config/dosentencepairs_template.json", configfilename;
		  pattern=pattern, dataset=datasetname,
                  pretrained_weight=pretrained_weight)
  # Run training
  println("Starting training")
  run(`python -m src.train -c $configfilename`)
  # Determine dir of generated data
  resultbasedir = joinpath("./exp_out/", datasetname, pretrained_weight)
  resultdir = sort(readdir(resultbasedir, join=true))[end]
  # Run test
  run(`python -m src.test -e $resultdir`)
  # Evaluate results
  accuracy = evaluate(resultdir, joinpath("./data", datasetname, "test.jsonl"))
  println("accuracy: $accuracy")

  # Unlock GPU
  unlock_gpu(gpu)
end

# ╔═╡ 168554b0-9886-477e-849d-e3aa8b503460
run_experiment(pattern=args["pattern"],
	       samples_per_aspect=args["samples_per_aspect"],
	       nb_neg_samples=args["nb_neg_samples"],
	       pretrained_weight=args["pretrained_weight"],
               seed=args["seed"],
               activelearningmetric=Symbol(args["use_active_learning"]),
               active_learning_threshold=args["active_learning_threshold"],
               act_shuffle=parse(Bool, args["act_shuffle"]),
               guided_sampling=parse(Bool, args["guided_sampling"]),
               )

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
MLDataPattern = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
ArgParse = "~1.1.4"
JSON = "~0.21.2"
MLDataPattern = "~0.5.4"
NPZ = "~0.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgParse]]
deps = ["Logging", "TextWrap"]
git-tree-sha1 = "3102bce13da501c9104df33549f511cd25264d7d"
uuid = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "79b9563ef3f2cc5fc6d3046a5ee1a57c9de52495"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.33.0"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "88d48e133e6d3dd68183309877eac74393daa7eb"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.17.20"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LearnBase]]
git-tree-sha1 = "a0d90569edd490b82fdc4dc078ea54a5a800d30a"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.4.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "3211c1fdd1efaefa692c8cf60e021fb007b76a08"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.6"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NPZ]]
deps = ["Compat", "ZipFile"]
git-tree-sha1 = "fbfb3c151b0308236d854c555b43cdd84c1e5ebf"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "477bf42b4d1496b454c10cce46645bb5b8a0cf2c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures", "Random", "Test"]
git-tree-sha1 = "03f5898c9959f8115e30bc7226ada7d0df554ddd"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "0.3.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TextWrap]]
git-tree-sha1 = "9250ef9b01b66667380cf3275b3f7488d0e25faf"
uuid = "b718987f-49a8-5099-9789-dcd902bef87d"
version = "1.0.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═5dbcd3de-fe92-11eb-01f7-fbe7e0714a6a
# ╠═190194c2-aea0-4053-85ed-b5d5d97632fb
# ╠═79098e1e-5917-484c-9324-757548f1f699
# ╠═d8903bd9-38d2-4b62-9e2c-f46e1ef7f196
# ╠═faad7cdd-2811-46c4-9387-c6934b4e1359
# ╠═ea852da8-2bee-4f54-86f2-fe1222d38361
# ╠═354d0e38-ebf3-4069-921e-90588ff8e78e
# ╠═b4587bec-c0fd-474f-9eb7-1a795c203452
# ╠═8e5fddff-33e8-4854-91d6-c8ee6895b3c5
# ╠═168554b0-9886-477e-849d-e3aa8b503460
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
